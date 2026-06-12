#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_scaling_report.py

Reads a directory with batch_X subdirectories (X = 32, 128, 256, ...) and
generates a scaling table showing:
  - throughput, latency per batch, latency per row
  - CPU and RAM utilization
for each BATCH × variant (cpu / hailo).

Usage:
  python3 gen_scaling_report.py \\
      --scaling-dir bench_results/scaling_20260509_xxxx \\
      --out scaling_report.md
"""

import argparse
import re
from pathlib import Path
import pandas as pd
import numpy as np


def fmt(x, prec=2, dash="—"):
    if x is None or pd.isna(x):
        return dash
    if isinstance(x, (int, np.integer)):
        return f"{x}"
    return f"{x:.{prec}f}"


def stats_for_dir(batch_dir: Path):
    """Calculate statistics for a batch_X directory."""
    out = {"cpu": {}, "hailo": {}}

    for variant in ("cpu", "hailo"):
        timing_path = batch_dir / f"timing_{variant}.csv"
        resources_path = batch_dir / f"resources_{variant}.csv"

        if timing_path.exists():
            t = pd.read_csv(timing_path)
            if not t.empty:
                total_rows = t["n_rows"].sum()
                total_time_s = t["t_total_ms"].sum() / 1000.0
                rows_per_s = total_rows / total_time_s if total_time_s > 0 else 0
                out[variant].update({
                    "n_batches": len(t),
                    "total_rows": total_rows,
                    "rows_per_s": rows_per_s,
                    "lat_per_inf_ms": t["t_inference_ms"].mean() / max(t["n_rows"].mean(), 1),
                    "batch_avg_ms": t["t_total_ms"].mean(),
                    "batch_p50_ms": t["t_total_ms"].quantile(0.5),
                    "batch_p95_ms": t["t_total_ms"].quantile(0.95),
                    "inf_avg_ms": t["t_inference_ms"].mean(),
                    "preprocess_avg_ms": t["t_preprocess_ms"].mean(),
                })

        if resources_path.exists():
            r = pd.read_csv(resources_path)
            if not r.empty and len(r) > 2:
                r = r.iloc[2:]  # skip warmup
                out[variant].update({
                    "cpu_proc_avg": r["cpu_proc_pct"].mean(),
                    "cpu_proc_max": r["cpu_proc_pct"].max(),
                    "rss_mb": r["rss_mb"].mean(),
                })

    return out


def make_scaling_table(rows_data, metric_key, label, prec=1, lower_better=False):
    """Build a pivot table batch × (cpu, hailo) with a single metric."""
    lines = [f"### {label}", ""]
    header = "| Batch size |" + "".join(f" CPU | Hailo |" for _ in [1])
    sep = "|---|" + "---:|---:|"
    lines.append(header)
    lines.append(sep)

    for batch_size, stats in sorted(rows_data.items()):
        cpu_v = stats["cpu"].get(metric_key)
        h_v = stats["hailo"].get(metric_key)

        # Mark the winner
        cpu_str = fmt(cpu_v, prec)
        h_str = fmt(h_v, prec)

        if cpu_v is not None and h_v is not None:
            if lower_better:
                if cpu_v < h_v:
                    cpu_str = f"**{cpu_str}**"
                elif h_v < cpu_v:
                    h_str = f"**{h_str}**"
            else:
                if cpu_v > h_v:
                    cpu_str = f"**{cpu_str}**"
                elif h_v > cpu_v:
                    h_str = f"**{h_str}**"

        lines.append(f"| {batch_size} | {cpu_str} | {h_str} |")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scaling-dir", required=True, help="Directory with batch_X subdirs")
    ap.add_argument("--out", default="scaling_report.md")
    args = ap.parse_args()

    base = Path(args.scaling_dir)
    if not base.exists():
        raise SystemExit(f"Directory does not exist: {base}")

    # Find batch_X subdirectories
    batch_dirs = {}
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r"batch_(\d+)", d.name)
        if m:
            batch_dirs[int(m.group(1))] = d

    if not batch_dirs:
        raise SystemExit(f"No batch_X directories found in {base}")

    print(f"[SCALE] Found {len(batch_dirs)} batch sizes: {sorted(batch_dirs.keys())}")

    # Calculate statistics for each
    rows_data = {}
    for batch_size, batch_dir in batch_dirs.items():
        rows_data[batch_size] = stats_for_dir(batch_dir)
        print(f"[SCALE] batch={batch_size}: "
              f"cpu_batches={rows_data[batch_size]['cpu'].get('n_batches', 0)}, "
              f"hailo_batches={rows_data[batch_size]['hailo'].get('n_batches', 0)}")

    # Build the report
    parts = [
        f"# Scaling Benchmark: CPU vs Hailo",
        f"",
        f"Directory: `{base}`",
        f"",
        f"Hardware: Raspberry Pi 5 + Hailo-8 (26 TOPs, M.2 PCIe)",
        f"",
        f"Batch sizes tested: {', '.join(str(b) for b in sorted(rows_data.keys()))}",
        f"",
        f"**Bold** = winner for the respective metric.",
        f"",
        f"---",
        f"",
        make_scaling_table(rows_data, "rows_per_s",
                           "Table S1 — Throughput (rows/sec, higher is better)", prec=1),
        make_scaling_table(rows_data, "batch_avg_ms",
                           "Table S2 — Average latency per batch (ms, lower is better)",
                           prec=2, lower_better=True),
        make_scaling_table(rows_data, "lat_per_inf_ms",
                           "Table S3 — Latency per inference (ms/row, lower is better)",
                           prec=4, lower_better=True),
        make_scaling_table(rows_data, "inf_avg_ms",
                           "Table S4 — Pure inference time per batch (ms, without preprocess/log)",
                           prec=3, lower_better=True),
        make_scaling_table(rows_data, "preprocess_avg_ms",
                           "Table S5 — Preprocess per batch (ms — dominates total!)",
                           prec=3, lower_better=True),
        make_scaling_table(rows_data, "batch_p95_ms",
                           "Table S6 — p95 latency per batch (ms, predictability)",
                           prec=2, lower_better=True),
        make_scaling_table(rows_data, "cpu_proc_avg",
                           "Table S7 — Average CPU (%, lower is better)",
                           prec=1, lower_better=True),
        make_scaling_table(rows_data, "rss_mb",
                           "Table S8 — RAM (MB, lower is better)",
                           prec=1, lower_better=True),
        f"---",
        f"",
        f"## How to read the tables",
        f"",
        f"- **Throughput (S1)**: number of rows / second processed end-to-end. ",
        f"  Includes preprocessing + inference + log. ",
        f"  For an IPS, represents the maximum flow throughput it can sustain.",
        f"- **Latency per batch (S2)**: how long a batch takes from `pd.read_csv` slice ",
        f"  to log written. Includes Hailo overhead (~3 ms / batch context activate).",
        f"- **Pure inference (S4)**: only the time for `runner.infer()` or `sess.run()`. ",
        f"  At large batch sizes, the fixed Hailo cost amortizes — see how the CPU column ",
        f"  approaches or is overtaken by Hailo.",
        f"- **Preprocess (S5)**: pandas + scaler. **Bottleneck** at small batch sizes. ",
        f"  Regardless of the ML backend used, this dominates the total time.",
        f"- **p95 (S6)**: 95% of batches finish below this value. ",
        f"  Important for SLA: shows the tail of the distribution, not just the average.",
        f"- **CPU & RAM (S7-S8)**: process utilization. ",
        f"  With Hailo, CPU usage is lower because computation migrates to the NPU.",
        f"",
        f"## Scaling conclusion",
        f"",
        f"For a small binary MLP, **Hailo does not win on throughput** ",
        f"except possibly at large batch sizes. The real gain is in **resource utilization** ",
        f"(CPU, RAM) and **predictability** (lower p95/p99). For an IPS system ",
        f"where ML is one of several components (capture, flow parsing, decision, ",
        f"iptables actuator), the value of Hailo is freeing up the CPU for the rest of the ",
        f"pipeline, not accelerating the ML itself.",
    ]

    out_path = Path(args.out)
    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"[DONE] Report: {out_path}")
    print()
    print("\n".join(parts))


if __name__ == "__main__":
    main()
