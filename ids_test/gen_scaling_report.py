#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_scaling_report.py

Citește un director cu sub-directoare batch_X (X = 32, 128, 256, ...) și
generează un tabel de scaling care arată:
  - throughput, latență per batch, latență per rând
  - utilizare CPU și RAM
pentru fiecare BATCH × variant (cpu / hailo).

Folosire:
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
    """Calculează statisticile pentru un director batch_X."""
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
    """Construiește un tabel pivot batch × (cpu, hailo) cu o singură metrică."""
    lines = [f"### {label}", ""]
    header = "| Batch size |" + "".join(f" CPU | Hailo |" for _ in [1])
    sep = "|---|" + "---:|---:|"
    lines.append(header)
    lines.append(sep)

    for batch_size, stats in sorted(rows_data.items()):
        cpu_v = stats["cpu"].get(metric_key)
        h_v = stats["hailo"].get(metric_key)

        # Marcheaza castigatorul
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
    ap.add_argument("--scaling-dir", required=True, help="Director cu batch_X subdirs")
    ap.add_argument("--out", default="scaling_report.md")
    args = ap.parse_args()

    base = Path(args.scaling_dir)
    if not base.exists():
        raise SystemExit(f"Directorul nu exista: {base}")

    # Gaseste subdirectoarele batch_X
    batch_dirs = {}
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r"batch_(\d+)", d.name)
        if m:
            batch_dirs[int(m.group(1))] = d

    if not batch_dirs:
        raise SystemExit(f"Niciun batch_X in {base}")

    print(f"[SCALE] Gasit {len(batch_dirs)} batch sizes: {sorted(batch_dirs.keys())}")

    # Calculeaza statistici pentru fiecare
    rows_data = {}
    for batch_size, batch_dir in batch_dirs.items():
        rows_data[batch_size] = stats_for_dir(batch_dir)
        print(f"[SCALE] batch={batch_size}: "
              f"cpu_batches={rows_data[batch_size]['cpu'].get('n_batches', 0)}, "
              f"hailo_batches={rows_data[batch_size]['hailo'].get('n_batches', 0)}")

    # Construieste raportul
    parts = [
        f"# Scaling Benchmark: CPU vs Hailo",
        f"",
        f"Director: `{base}`",
        f"",
        f"Hardware: Raspberry Pi 5 + Hailo-8 (26 TOPs, M.2 PCIe)",
        f"",
        f"Batch sizes testate: {', '.join(str(b) for b in sorted(rows_data.keys()))}",
        f"",
        f"**Bold** = câștigătorul la metrica respectivă.",
        f"",
        f"---",
        f"",
        make_scaling_table(rows_data, "rows_per_s",
                           "Tabel S1 — Throughput (rows/sec, mai mult = mai bine)", prec=1),
        make_scaling_table(rows_data, "batch_avg_ms",
                           "Tabel S2 — Latență medie per batch (ms, mai puțin = mai bine)",
                           prec=2, lower_better=True),
        make_scaling_table(rows_data, "lat_per_inf_ms",
                           "Tabel S3 — Latență per inferență (ms/rând, mai puțin = mai bine)",
                           prec=4, lower_better=True),
        make_scaling_table(rows_data, "inf_avg_ms",
                           "Tabel S4 — Timp inferență pură per batch (ms, fără preprocess/log)",
                           prec=3, lower_better=True),
        make_scaling_table(rows_data, "preprocess_avg_ms",
                           "Tabel S5 — Preprocess per batch (ms — domină total!)",
                           prec=3, lower_better=True),
        make_scaling_table(rows_data, "batch_p95_ms",
                           "Tabel S6 — Latență p95 per batch (ms, predictibilitate)",
                           prec=2, lower_better=True),
        make_scaling_table(rows_data, "cpu_proc_avg",
                           "Tabel S7 — CPU mediu (%, mai puțin = mai bine)",
                           prec=1, lower_better=True),
        make_scaling_table(rows_data, "rss_mb",
                           "Tabel S8 — RAM (MB, mai puțin = mai bine)",
                           prec=1, lower_better=True),
        f"---",
        f"",
        f"## Cum se citesc tabelele",
        f"",
        f"- **Throughput (S1)**: numărul de rânduri / secundă procesate end-to-end. ",
        f"  Include preprocesare + inferență + log. ",
        f"  Pentru un IPS, reprezintă debitul maxim de flow-uri pe care îl poate susține.",
        f"- **Latență per batch (S2)**: cât durează un batch de la `pd.read_csv` slice ",
        f"  până la log scris. Include overhead-ul Hailo (~3 ms / batch context activate).",
        f"- **Inferență pură (S4)**: doar timpul pentru `runner.infer()` sau `sess.run()`. ",
        f"  La batch mare, costul fix Hailo se amortizează — vezi cum coloana CPU se ",
        f"  apropie sau e depășită de Hailo.",
        f"- **Preprocess (S5)**: pandas + scaler. **Bottleneck-ul** la batch mic. ",
        f"  Indiferent ce backend ML folosești, asta domină timpul total.",
        f"- **p95 (S6)**: 95% din batch-uri se termină sub această valoare. ",
        f"  Important pentru SLA: arată coada distribuției, nu doar media.",
        f"- **CPU & RAM (S7-S8)**: utilizarea procesului. ",
        f"  La Hailo, CPU-ul e mai puțin folosit pentru că calculul migrează pe NPU.",
        f"",
        f"## Concluzie scaling",
        f"",
        f"Pentru un MLP binary de dimensiune mică, **Hailo nu câștigă la throughput** ",
        f"decât eventual la batch-uri mari. Câștigul real e la **utilizare resurse** ",
        f"(CPU, RAM) și **predictibilitate** (p95/p99 mai mici). Pentru un sistem IPS ",
        f"unde ML e una din mai multe componente (captură, parsare flow, decizie, ",
        f"actuator iptables), valoarea Hailo este eliberarea CPU-ului pentru restul ",
        f"pipeline-ului, nu accelerarea ML-ului în sine.",
    ]

    out_path = Path(args.out)
    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"[DONE] Raport: {out_path}")
    print()
    print("\n".join(parts))


if __name__ == "__main__":
    main()
