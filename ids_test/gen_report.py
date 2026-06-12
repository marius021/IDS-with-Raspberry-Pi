#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_report.py

Reads all CSVs from a benchmark directory and generates
a Markdown report with tables for documentation.

Expects in --bench-dir:
  timing_cpu.csv
  timing_hailo.csv
  resources_cpu.csv
  resources_hailo.csv

Usage:
  python3 gen_report.py --bench-dir bench_results/20260509_180000 --out report.md
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np


def fmt(x, prec=2):
    if pd.isna(x) or x == "":
        return "—"
    if isinstance(x, (int, np.integer)):
        return f"{x}"
    return f"{x:.{prec}f}"


def load_timing(path: Path):
    if not path.exists():
        print(f"[WARN] Missing {path}")
        return None
    df = pd.read_csv(path)
    return df


def load_resources(path: Path):
    if not path.exists():
        print(f"[WARN] Missing {path}")
        return None
    df = pd.read_csv(path)
    return df


def stage_latency_table(t_cpu, t_hailo):
    """Table 1: latency per stage (ms per batch, average)."""
    stages = ["read", "preprocess", "inference", "postprocess", "log"]
    lines = ["### Table 1 — Latency per stage (ms / batch, average values)",
             "",
             "| Stage | CPU (ONNX) | Hailo (HEF) | Hailo / CPU |",
             "|---|---:|---:|---:|"]
    for s in stages:
        col = f"t_{s}_ms"
        cpu_v = t_cpu[col].mean() if (t_cpu is not None and col in t_cpu.columns) else None
        h_v = t_hailo[col].mean() if (t_hailo is not None and col in t_hailo.columns) else None
        ratio = (h_v / cpu_v) if (cpu_v and cpu_v > 0 and h_v is not None) else None
        lines.append(f"| {s} | {fmt(cpu_v, 3)} | {fmt(h_v, 3)} | {fmt(ratio, 2) if ratio else '—'} |")

    cpu_total = t_cpu["t_total_ms"].mean() if t_cpu is not None else None
    h_total = t_hailo["t_total_ms"].mean() if t_hailo is not None else None
    total_ratio = (h_total / cpu_total) if (cpu_total and cpu_total > 0 and h_total) else None
    lines.append(f"| **TOTAL** | **{fmt(cpu_total, 3)}** | **{fmt(h_total, 3)}** | **{fmt(total_ratio, 2) if total_ratio else '—'}** |")
    return "\n".join(lines)


def throughput_table(t_cpu, t_hailo):
    """Table 2: end-to-end throughput."""
    lines = ["### Table 2 — End-to-end throughput",
             "",
             "| Metric | CPU (ONNX) | Hailo (HEF) |",
             "|---|---:|---:|"]

    def stats(df):
        if df is None or df.empty:
            return {}
        total_rows = df["n_rows"].sum()
        total_time_s = df["t_total_ms"].sum() / 1000.0
        rows_per_s = total_rows / total_time_s if total_time_s > 0 else 0
        avg_lat_per_inf_ms = df["t_inference_ms"].mean() / df["n_rows"].mean()
        avg_batch_lat_ms = df["t_total_ms"].mean()
        p50 = df["t_total_ms"].quantile(0.5)
        p95 = df["t_total_ms"].quantile(0.95)
        p99 = df["t_total_ms"].quantile(0.99)
        return {
            "rows_per_s": rows_per_s,
            "avg_lat_per_inf_ms": avg_lat_per_inf_ms,
            "avg_batch_lat_ms": avg_batch_lat_ms,
            "p50_batch_ms": p50,
            "p95_batch_ms": p95,
            "p99_batch_ms": p99,
            "n_batches": len(df),
            "total_rows": total_rows,
        }

    sc = stats(t_cpu)
    sh = stats(t_hailo)

    rows = [
        ("Total rows processed", "total_rows", 0),
        ("Total batches", "n_batches", 0),
        ("Throughput (rows/s)", "rows_per_s", 1),
        ("Average latency per row (ms)", "avg_lat_per_inf_ms", 4),
        ("Average latency per batch (ms)", "avg_batch_lat_ms", 3),
        ("p50 latency per batch (ms)", "p50_batch_ms", 3),
        ("p95 latency per batch (ms)", "p95_batch_ms", 3),
        ("p99 latency per batch (ms)", "p99_batch_ms", 3),
    ]
    for label, key, prec in rows:
        cpu_v = sc.get(key)
        h_v = sh.get(key)
        lines.append(f"| {label} | {fmt(cpu_v, prec)} | {fmt(h_v, prec)} |")

    return "\n".join(lines)


def resources_table(r_cpu, r_hailo):
    """Table 3: system resources in steady state."""
    lines = ["### Table 3 — Resource utilization (steady state)",
             "",
             "| Metric | CPU (ONNX) | Hailo (HEF) |",
             "|---|---:|---:|"]

    def stats(df):
        if df is None or df.empty:
            return {}
        # Skip first 2 samples (warm-up)
        df = df.iloc[2:] if len(df) > 2 else df
        return {
            "cpu_total_avg": df["cpu_total_pct"].mean(),
            "cpu_total_max": df["cpu_total_pct"].max(),
            "cpu_proc_avg": df["cpu_proc_pct"].mean(),
            "cpu_proc_max": df["cpu_proc_pct"].max(),
            "rss_avg_mb": df["rss_mb"].mean(),
            "rss_max_mb": df["rss_mb"].max(),
            "temp_avg": df["temp_c"].mean() if "temp_c" in df.columns else None,
            "temp_max": df["temp_c"].max() if "temp_c" in df.columns else None,
            "n_samples": len(df),
        }

    sc = stats(r_cpu)
    sh = stats(r_hailo)

    rows = [
        ("CPU total avg (%)", "cpu_total_avg", 1),
        ("CPU total peak (%)", "cpu_total_max", 1),
        ("CPU process avg (%)", "cpu_proc_avg", 1),
        ("CPU process peak (%)", "cpu_proc_max", 1),
        ("RAM (RSS) avg (MB)", "rss_avg_mb", 1),
        ("RAM (RSS) peak (MB)", "rss_max_mb", 1),
        ("Temperature avg (°C)", "temp_avg", 1),
        ("Temperature peak (°C)", "temp_max", 1),
    ]
    for label, key, prec in rows:
        lines.append(f"| {label} | {fmt(sc.get(key), prec)} | {fmt(sh.get(key), prec)} |")

    return "\n".join(lines)


def per_core_table(r_cpu, r_hailo):
    """Table 4: %CPU per core (Pi 5 has 4 cores)."""
    lines = ["### Table 4 — %CPU distribution per core (avg)",
             "",
             "| Core | CPU (ONNX) | Hailo (HEF) |",
             "|---|---:|---:|"]

    def core_avgs(df):
        if df is None or df.empty:
            return {}
        df = df.iloc[2:] if len(df) > 2 else df
        cores = [c for c in df.columns if c.startswith("core") and c.endswith("_pct")]
        return {c: df[c].mean() for c in sorted(cores)}

    cc = core_avgs(r_cpu)
    ch = core_avgs(r_hailo)

    all_cores = sorted(set(list(cc.keys()) + list(ch.keys())))
    for core in all_cores:
        lines.append(f"| {core} | {fmt(cc.get(core), 1)} | {fmt(ch.get(core), 1)} |")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", required=True, help="Directory with benchmark CSVs")
    ap.add_argument("--out", default="report.md", help="Markdown report file")
    args = ap.parse_args()

    base = Path(args.bench_dir)
    if not base.exists():
        sys.exit(f"Directory does not exist: {base}")

    t_cpu = load_timing(base / "timing_cpu.csv")
    t_hailo = load_timing(base / "timing_hailo.csv")
    r_cpu = load_resources(base / "resources_cpu.csv")
    r_hailo = load_resources(base / "resources_hailo.csv")

    parts = [
        f"# Benchmark IPS: CPU vs Hailo",
        f"",
        f"Directory: `{base}`",
        f"",
        f"Hardware: Raspberry Pi 5 + Hailo-8 (26 TOPs, M.2 PCIe)",
        f"",
        f"---",
        f"",
        stage_latency_table(t_cpu, t_hailo),
        f"",
        throughput_table(t_cpu, t_hailo),
        f"",
        resources_table(r_cpu, r_hailo),
        f"",
        per_core_table(r_cpu, r_hailo),
        f"",
        f"---",
        f"",
        f"## Interpretation notes",
        f"",
        f"- **Latency per stage**: includes the Hailo context activation overhead per batch ",
        f"  (~10-50 ms typical). For a real-time IPS this is acceptable; for pure throughput ",
        f"  it could be reduced by keeping the context active between batches.",
        f"- **CPU process**: on the CPU (ONNX) variant, the process saturates 1 core (~100% on one core ",
        f"  means ~25% on a Pi with 4 cores). On Hailo, computation is offloaded → CPU sits idle.",
        f"- **RAM**: similar on both variants (the model is small), most consumption is Python ",
        f"  + pandas + onnxruntime/HailoRT.",
    ]

    out_path = Path(args.out)
    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"[REPORT] Written to: {out_path}")
    print()
    # Print report to stdout
    print("\n".join(parts))


if __name__ == "__main__":
    main()
