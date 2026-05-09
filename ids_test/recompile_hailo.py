#!/usr/bin/env python3
"""
recompile_hailo.py

Recompiles ids_mlp_binary_logits.onnx → HEF using calibration data that
exactly matches what the model receives at runtime (StandardScaler-normalized
float32 inputs).  Run on the Dell inside hailo_env.

Root cause this fixes:
  The previous HEF was quantized with calibration data that did not match
  runtime inputs, causing all logits to be extremely negative (~-35 mean)
  and the model to classify every flow as BENIGN.

Usage:
  source hailo_env/bin/activate
  cd ids_test
  python3 recompile_hailo.py

  # Custom calibration size or paths:
  python3 recompile_hailo.py --calib-csv ../ids_test/sample_labeled.csv --n-calib 512
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── Preprocessing (must match ips_hailo.py exactly) ─────────────────────────

def load_scaler_params(npz_path: Path) -> dict:
    data = np.load(npz_path)
    return {
        "mean": data["mean"].astype(np.float32),
        "scale": data["scale"].astype(np.float32),
        "with_mean": bool(int(data["with_mean"][0])),
        "with_std": bool(int(data["with_std"][0])),
        "n_features_in": int(data["n_features_in"][0]),
    }


def apply_scaler(X: np.ndarray, sp: dict) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32).copy()
    if sp["with_mean"]:
        X -= sp["mean"]
    if sp["with_std"]:
        scale = sp["scale"].copy()
        scale[scale == 0] = 1.0
        X /= scale
    return np.ascontiguousarray(X, dtype=np.float32)


def load_feature_names(path: Path):
    return [l.strip().lower() for l in
            path.read_text(encoding="utf-8").splitlines() if l.strip()]


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns.astype(str)
                  .str.replace("﻿", "", regex=False)
                  .str.strip().str.lower())
    return df


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan).clip(-1e12, 1e12)
    return df.fillna(df.median(numeric_only=True)).fillna(0)


def build_calib_data(csv_path: Path, sp: dict, feats: list,
                     n: int, seed: int) -> np.ndarray:
    """Return float32 array of shape (n, 1, 1, 80) — the Hailo input shape."""
    df = pd.read_csv(csv_path, low_memory=False)
    df = normalize_cols(df)
    df_num = df.select_dtypes(include=[np.number]).copy()
    df_num = clean_numeric(df_num)

    missing = [c for c in feats if c not in df_num.columns]
    if missing:
        raise RuntimeError(f"Missing columns in calibration CSV: {missing[:10]}")

    df_num = df_num[feats]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df_num), size=min(n, len(df_num)), replace=False)
    X = df_num.iloc[idx].values.astype(np.float32)
    Xs = apply_scaler(X, sp)

    # Hailo calibration shape: (N, C, H, W) → (N, 1, 1, n_features)
    return Xs.reshape(len(Xs), 1, 1, sp["n_features_in"])


# ── Hailo compilation pipeline ───────────────────────────────────────────────

def compile_hef(onnx_path: Path, calib: np.ndarray,
                hw_arch: str, out_hef: Path) -> None:
    from hailo_sdk_client import ClientRunner

    print(f"\n[1/3] Parsing ONNX → HAR  ({onnx_path.name})")
    runner = ClientRunner(hw_arch=hw_arch)
    runner.translate_onnx_model(
        str(onnx_path),
        onnx_path.stem,
    )
    har_parsed = out_hef.with_suffix("").with_suffix(".parsed.har")
    runner.save_har(str(har_parsed))
    print(f"      Saved: {har_parsed}")

    print(f"\n[2/3] Optimizing (quantizing) with {len(calib)} calibration samples")
    print(f"      Calibration data: shape={calib.shape} "
          f"min={calib.min():.3f} max={calib.max():.3f} mean={calib.mean():.3f}")
    runner = ClientRunner(hw_arch=hw_arch, har=str(har_parsed))
    runner.optimize(calib)
    har_quant = out_hef.with_suffix("").with_suffix(".quant.har")
    runner.save_har(str(har_quant))
    print(f"      Saved: {har_quant}")

    print(f"\n[3/3] Compiling → HEF")
    runner = ClientRunner(hw_arch=hw_arch, har=str(har_quant))
    hef_bytes = runner.compile()
    out_hef.write_bytes(hef_bytes)
    print(f"      Saved: {out_hef}  ({out_hef.stat().st_size / 1024:.0f} KB)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Recompile Hailo HEF with correctly preprocessed calibration data."
    )
    ap.add_argument("--onnx", default="ids_mlp_binary_logits.onnx",
                    help="Source ONNX model (logits variant)")
    ap.add_argument("--scaler", default="scaler_params.npz")
    ap.add_argument("--features", default="feature_names.txt")
    ap.add_argument("--calib-csv", default="sample_labeled.csv",
                    help="CSV used to build calibration samples (must have all feature columns)")
    ap.add_argument("--n-calib", type=int, default=512,
                    help="Number of calibration samples (default 512)")
    ap.add_argument("--hw-arch", default="hailo8",
                    help="Hailo hardware architecture (hailo8 or hailo8l)")
    ap.add_argument("--out", default="ids_mlp_binary_logits.hef",
                    help="Output HEF path (overwrites existing)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    scaler_path = Path(args.scaler)
    feats_path = Path(args.features)
    calib_csv = Path(args.calib_csv)
    out_hef = Path(args.out)

    for label, p in [("ONNX", onnx_path), ("Scaler", scaler_path),
                     ("Features", feats_path), ("Calib CSV", calib_csv)]:
        if not p.exists():
            sys.exit(f"[ERROR] {label} not found: {p}")

    print("=" * 60)
    print("  HAILO HEF RECOMPILATION")
    print("=" * 60)
    print(f"  ONNX     : {onnx_path}")
    print(f"  Calib CSV: {calib_csv}  (n={args.n_calib})")
    print(f"  HW arch  : {args.hw_arch}")
    print(f"  Output   : {out_hef}")

    sp = load_scaler_params(scaler_path)
    feats = load_feature_names(feats_path)

    print(f"\n[0/3] Building calibration data from {calib_csv.name}...")
    calib = build_calib_data(calib_csv, sp, feats, args.n_calib, args.seed)
    print(f"      Shape : {calib.shape}")
    print(f"      Range : [{calib.min():.3f}, {calib.max():.3f}]")
    print(f"      Mean  : {calib.mean():.3f}  Std: {calib.std():.3f}")

    compile_hef(onnx_path, calib, args.hw_arch, out_hef)

    print(f"\n{'=' * 60}")
    print(f"  Done. Copy {out_hef} to the Raspberry Pi and retest.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
