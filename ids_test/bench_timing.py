#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_timing.py

Small instrumentation module for the IPS pipeline.
Activated only if the environment variable BENCH=1 or BENCH_OUT is set.

Minimal-invasive usage (see instructions_apply.md):

    from bench_timing import StageTimer, BenchWriter, maybe_writer

    bench = maybe_writer()      # None if not running in benchmark mode
    timer = StageTimer()
    batch_idx = 0

    timer.reset()
    timer.start("read");       df = pd.read_csv(...);                timer.stop("read")
    timer.start("preprocess"); Xs = build_feature_matrix(...);       timer.stop("preprocess")
    timer.start("inference");  prob, pred = run_batch(...);          timer.stop("inference")
    timer.start("postprocess");...sigmoid/threshold...;              timer.stop("postprocess")
    timer.start("log");        append_alerts/append_actions(...);    timer.stop("log")

    if bench:
        bench.write_batch(batch_idx, n_rows=len(df), timer=timer)
        batch_idx += 1
"""

import os
import time
import csv
from pathlib import Path
from typing import Dict, List, Optional


class StageTimer:
    """Accumulates times per stage for a single batch."""

    def __init__(self):
        self._starts: Dict[str, float] = {}
        self.times: Dict[str, float] = {}

    def start(self, stage: str):
        self._starts[stage] = time.perf_counter()

    def stop(self, stage: str):
        if stage not in self._starts:
            raise RuntimeError(f"start('{stage}') was not called before stop()")
        elapsed_ms = (time.perf_counter() - self._starts[stage]) * 1000.0
        # Accumulate if the same stage is timed more than once per batch
        self.times[stage] = self.times.get(stage, 0.0) + elapsed_ms

    def reset(self):
        self._starts.clear()
        self.times.clear()

    def total_ms(self) -> float:
        return sum(self.times.values())


class BenchWriter:
    """CSV writer for per-batch timing. One line = one batch."""

    DEFAULT_STAGES = ["read", "preprocess", "inference", "postprocess", "log"]

    def __init__(self, path: str, stages: Optional[List[str]] = None,
                 variant: str = "unknown"):
        self.path = Path(path)
        self.stages = stages or self.DEFAULT_STAGES
        self.variant = variant
        self.path.parent.mkdir(parents=True, exist_ok=True)

        first_write = not self.path.exists()
        self._fh = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)

        if first_write:
            cols = ["timestamp", "variant", "batch_idx", "n_rows"] + \
                   [f"t_{s}_ms" for s in self.stages] + ["t_total_ms"]
            self._writer.writerow(cols)
            self._fh.flush()

    def write_batch(self, batch_idx: int, n_rows: int, timer: StageTimer):
        row = [time.time(), self.variant, batch_idx, n_rows]
        for s in self.stages:
            row.append(round(timer.times.get(s, 0.0), 4))
        row.append(round(timer.total_ms(), 4))
        self._writer.writerow(row)
        self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def maybe_writer(default_path: str = "timing.csv",
                 default_variant: str = "unknown") -> Optional[BenchWriter]:
    """
    Returns a BenchWriter if BENCH=1 or BENCH_OUT is set.
    Returns None otherwise — so scripts run normally without overhead.

    Environment variables:
      BENCH       = 1 / 0  (enable / disable)
      BENCH_OUT   = CSV file path (overrides default_path)
      BENCH_VARIANT = label (cpu / hailo / etc)
    """
    if not (os.getenv("BENCH") == "1" or os.getenv("BENCH_OUT")):
        return None
    out = os.getenv("BENCH_OUT", default_path)
    variant = os.getenv("BENCH_VARIANT", default_variant)
    print(f"[BENCH] Timing enabled. Writing to: {out} (variant={variant})")
    return BenchWriter(out, variant=variant)
