#!/usr/bin/env python3
"""
validate_hailo.py

End-to-end validation of the Hailo HEF model on a labeled CSV.
Runs on Raspberry Pi with Hailo hardware attached.

Steps performed:
  1. Load scaler_params.npz + feature names
  2. Preprocess the labeled CSV identically to ips_hailo.py
  3. Run Hailo inference in batches
  4. Print raw output diagnostics (logits vs probabilities)
  5. Compute precision / recall / F1 at the chosen threshold
  6. Optionally sweep thresholds to find the best F1
  7. Optionally compare against the ONNX model on the same inputs
  8. Save a JSON report

Usage examples:
  # Basic run
  python3 validate_hailo.py \\
      --csv sample.csv \\
      --hef ids_mlp_binary_logits.hef \\
      --scaler scaler_params.npz \\
      --features feature_names.txt

  # Full run with threshold sweep + ONNX comparison + report
  python3 validate_hailo.py \\
      --csv sample.csv \\
      --hef ids_mlp_binary_logits.hef \\
      --scaler scaler_params.npz \\
      --features feature_names.txt \\
      --sweep \\
      --onnx ids_mlp_binary.onnx \\
      --out validation_report.json
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


# ── Scaler ──────────────────────────────────────────────────────────────────

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


# ── Feature names ────────────────────────────────────────────────────────────

def load_feature_names(path: Path) -> List[str]:
    suf = path.suffix.lower()
    if suf in (".txt", ".lst"):
        return [l.strip().lower() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if suf == ".npy":
        arr = np.load(path, allow_pickle=True)
        return [str(x).strip().lower() for x in arr.tolist()]
    if suf == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        lst = obj if isinstance(obj, list) else obj.get("features", [])
        return [str(x).strip().lower() for x in lst]
    raise RuntimeError(f"Unsupported feature file format: {path.suffix}")


# ── Preprocessing ────────────────────────────────────────────────────────────

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("﻿", "", regex=False)
        .str.strip()
        .str.lower()
    )
    return df


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.clip(-1e12, 1e12)
    med = df.median(numeric_only=True)
    return df.fillna(med).fillna(0)


def build_X(df: pd.DataFrame, sp: dict, feats: List[str]) -> np.ndarray:
    df = normalize_cols(df)
    df_num = df.select_dtypes(include=[np.number]).copy()
    df_num = clean_numeric(df_num)

    missing = [c for c in feats if c not in df_num.columns]
    if missing:
        raise RuntimeError(f"Missing columns vs training set: {', '.join(missing[:15])}")

    X = df_num[feats].values.astype(np.float32)
    if X.shape[1] != sp["n_features_in"]:
        raise RuntimeError(
            f"Feature count mismatch: scaler expects {sp['n_features_in']}, got {X.shape[1]}"
        )
    return apply_scaler(X, sp)


# ── Hailo runner ─────────────────────────────────────────────────────────────

def load_hailo_runner(hef_path: Path) -> dict:
    from hailo_platform import (
        HEF, VDevice, ConfigureParams, HailoStreamInterface,
        InputVStreamParams, OutputVStreamParams, FormatType,
    )
    hef = HEF(str(hef_path))
    device = VDevice()
    cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    configured = device.configure(hef, cfg)
    ng = configured[0] if isinstance(configured, (list, tuple)) else configured
    ng_params = ng.create_params()

    try:
        in_params = InputVStreamParams.make_from_network_group(
            ng, quantized=False, format_type=FormatType.FLOAT32
        )
        out_params = OutputVStreamParams.make_from_network_group(
            ng, quantized=False, format_type=FormatType.FLOAT32
        )
    except Exception:
        in_params = InputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
        out_params = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)

    in_info = hef.get_input_vstream_infos()[0]
    out_info = hef.get_output_vstream_infos()[0]
    n_feats = int(np.prod(tuple(in_info.shape))) if in_info.shape else 1

    return {
        "device": device,   # keep VDevice alive for the lifetime of inference
        "hef": hef,         # keep HEF alive too
        "ng": ng,
        "ng_params": ng_params,
        "in_params": in_params,
        "out_params": out_params,
        "in_name": in_info.name,
        "out_name": out_info.name,
        "in_shape": tuple(in_info.shape),
        "out_shape": tuple(out_info.shape),
        "n_features": n_feats,
    }


def hailo_infer_batch(runner: dict, Xs: np.ndarray) -> np.ndarray:
    from hailo_platform import InferVStreams
    Xs = np.ascontiguousarray(Xs, dtype=np.float32)
    single_shape = (1,) + runner["in_shape"]
    outputs = []
    with runner["ng"].activate(runner["ng_params"]):
        with InferVStreams(runner["ng"], runner["in_params"], runner["out_params"]) as pipe:
            for i in range(len(Xs)):
                x = Xs[i : i + 1].reshape(single_shape)
                res = pipe.infer({runner["in_name"]: x})
                out = np.asarray(res[runner["out_name"]])
                outputs.append(float(out.ravel()[0]))
    return np.array(outputs, dtype=np.float32)


# ── Math ─────────────────────────────────────────────────────────────────────

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


# ── Metrics (no sklearn dependency) ──────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    acc = (tp + tn) / max(len(y_true), 1)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }


def print_metrics(m: dict, indent: str = "  "):
    print(f"{indent}Accuracy : {m['accuracy']:.4f}")
    print(f"{indent}Precision: {m['precision']:.4f}  (TP={m['tp']}, FP={m['fp']})")
    print(f"{indent}Recall   : {m['recall']:.4f}  (TP={m['tp']}, FN={m['fn']})")
    print(f"{indent}F1       : {m['f1']:.4f}")
    print(f"{indent}Confusion: TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")


# ── Label helpers ─────────────────────────────────────────────────────────────

def detect_label_col(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    for cand in ("label", "class", "attack", "category"):
        if cand in cols:
            return cand
    return None


def extract_ground_truth(df: pd.DataFrame, label_col: str) -> np.ndarray:
    return (df[label_col].astype(str).str.strip().str.upper() != "BENIGN").astype(int).values


# ── ONNX helper ───────────────────────────────────────────────────────────────

def run_onnx_inference(onnx_path: Path, Xs: np.ndarray) -> np.ndarray:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    return sess.run(None, {in_name: Xs})[0].ravel().astype(np.float32)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Validate Hailo HEF model on a labeled CSV (run on Raspberry Pi).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--csv", required=True, help="Labeled CSV file (must have a Label column)")
    ap.add_argument("--hef", required=True, help="Hailo HEF model file")
    ap.add_argument("--scaler", required=True, help="scaler_params.npz")
    ap.add_argument("--features", required=True, help="Feature names file (.txt / .npy / .json)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Binary classification threshold (default 0.5)")
    ap.add_argument("--no-sigmoid", action="store_true",
                    help="Skip sigmoid — use if HEF already outputs probabilities [0,1]")
    ap.add_argument("--sweep", action="store_true",
                    help="Sweep thresholds and print F1 curve to find optimal threshold")
    ap.add_argument("--onnx", default="",
                    help="Optional ONNX model (.onnx) for side-by-side comparison")
    ap.add_argument("--batch", type=int, default=512, help="Hailo inference batch size")
    ap.add_argument("--out", default="",
                    help="Path to save JSON validation report (optional)")
    args = ap.parse_args()

    hef_path = Path(args.hef)
    scaler_path = Path(args.scaler)
    feats_path = Path(args.features)
    csv_path = Path(args.csv)

    for label, p in [("CSV", csv_path), ("HEF", hef_path),
                     ("Scaler", scaler_path), ("Features", feats_path)]:
        if not p.exists():
            sys.exit(f"[ERROR] {label} not found: {p}")

    print(f"\n{'='*60}")
    print("  HAILO END-TO-END VALIDATION")
    print(f"{'='*60}")
    print(f"  CSV       : {csv_path}")
    print(f"  HEF       : {hef_path}")
    print(f"  Scaler    : {scaler_path}")
    print(f"  Features  : {feats_path}")
    print(f"  Threshold : {args.threshold}")
    print(f"  Sigmoid   : {'disabled (--no-sigmoid)' if args.no_sigmoid else 'enabled'}")
    if args.onnx:
        print(f"  ONNX      : {args.onnx}")

    # ── Load artifacts ────────────────────────────────────────────────────────
    print("\n[1/5] Loading artifacts...")
    sp = load_scaler_params(scaler_path)
    feats = load_feature_names(feats_path)
    print(f"      Scaler : n_features_in={sp['n_features_in']}, "
          f"with_mean={sp['with_mean']}, with_std={sp['with_std']}")
    print(f"      Features: {len(feats)} names loaded")
    if sp["n_features_in"] != len(feats):
        print(f"      [WARN] Mismatch: scaler expects {sp['n_features_in']} "
              f"but feature list has {len(feats)}")

    # ── Load and preprocess CSV ───────────────────────────────────────────────
    print("\n[2/5] Loading and preprocessing CSV...")
    df_raw = pd.read_csv(csv_path, low_memory=False)
    df = normalize_cols(df_raw)
    print(f"      {len(df)} rows, {len(df.columns)} columns")

    label_col = detect_label_col(df)
    has_labels = label_col is not None
    if has_labels:
        y_true = extract_ground_truth(df, label_col)
        n_att = int(y_true.sum())
        print(f"      Label column: '{label_col}' | "
              f"ATTACK={n_att}, BENIGN={len(y_true)-n_att}")
    else:
        y_true = None
        print("      [WARN] No label column found — metrics will not be computed")

    Xs = build_X(df, sp, feats)
    print(f"      Feature matrix: {Xs.shape} | all finite: {np.isfinite(Xs).all()}")

    report = {
        "csv": str(csv_path),
        "hef": str(hef_path),
        "n_rows": len(Xs),
        "threshold": args.threshold,
        "sigmoid_applied": not args.no_sigmoid,
    }

    # ── Load Hailo (optional — skipped gracefully if not on Raspberry Pi) ────
    hailo_ok = False
    proba = None

    print("\n[3/5] Loading Hailo HEF...")
    t0 = time.time()
    try:
        runner = load_hailo_runner(hef_path)
        print(f"      Loaded in {time.time()-t0:.2f}s")
        print(f"      Input : {runner['in_name']} shape={runner['in_shape']} "
              f"({runner['n_features']} features)")
        print(f"      Output: {runner['out_name']} shape={runner['out_shape']}")
        if runner["n_features"] != Xs.shape[1]:
            print(f"      [WARN] HEF expects {runner['n_features']} features, "
                  f"data has {Xs.shape[1]}")

        # ── Run Hailo inference ───────────────────────────────────────────────
        print(f"\n[4/5] Running Hailo inference (batch={args.batch})...")
        t0 = time.time()
        raw_parts = []
        for i in range(0, len(Xs), args.batch):
            raw_parts.append(hailo_infer_batch(runner, Xs[i:i + args.batch]))
        raw_out = np.concatenate(raw_parts)
        elapsed = time.time() - t0
        tput = len(Xs) / elapsed if elapsed > 0 else 0

        print(f"      Done in {elapsed:.3f}s  ({tput:.0f} rows/s)")
        print(f"\n      Raw output diagnostics:")
        print(f"        min  = {raw_out.min():.6f}")
        print(f"        max  = {raw_out.max():.6f}")
        print(f"        mean = {raw_out.mean():.6f}")
        print(f"        std  = {raw_out.std():.6f}")
        if raw_out.min() < 0 or raw_out.max() > 1:
            print("        → Values outside [0,1]: model outputs LOGITS (sigmoid will be applied)")
        else:
            print("        → Values in [0,1]: model outputs PROBABILITIES")

        proba = raw_out.copy() if args.no_sigmoid else sigmoid(raw_out)
        if not args.no_sigmoid:
            print(f"\n      After sigmoid: min={proba.min():.6f} max={proba.max():.6f} "
                  f"mean={proba.mean():.6f}")

        report.update({
            "raw_out": {
                "min": float(raw_out.min()), "max": float(raw_out.max()),
                "mean": float(raw_out.mean()), "std": float(raw_out.std()),
            },
            "inference_sec": round(elapsed, 3),
            "rows_per_sec": round(tput, 1),
        })
        hailo_ok = True

    except Exception as e:
        if "hailo_platform" in str(e) or "No module" in str(e):
            print(f"      [WARN] hailo_platform not available — skipping Hailo inference.")
            print(f"             Run on Raspberry Pi for the full Hailo test.")
            print(f"\n[4/5] Hailo inference skipped.")
        else:
            print(f"      [ERROR] {type(e).__name__}: {e}")
            print(traceback.format_exc())
            print(f"\n[4/5] Hailo inference failed.")

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"\n[5/5] Results")

    def _run_metrics_and_sweep(proba_arr, label):
        if not has_labels:
            det = int((proba_arr >= args.threshold).sum())
            print(f"  No label column — rows classified as ATTACK @ {args.threshold}: "
                  f"{det}/{len(proba_arr)}")
            return

        pred = (proba_arr >= args.threshold).astype(int)
        print(f"\n  {label} @ threshold={args.threshold}:")
        m = compute_metrics(y_true, pred)
        print_metrics(m)
        report[f"{label.lower().split()[0]}_metrics"] = m

        if args.sweep:
            thresholds = [0.001, 0.005, 0.01, 0.02, 0.05,
                          0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.90]
            sweep = []
            for thr in thresholds:
                p_ = (proba_arr >= thr).astype(int)
                sweep.append({"threshold": thr, **compute_metrics(y_true, p_)})
            best_thr = max(sweep, key=lambda x: x["f1"])["threshold"]

            print(f"\n  Threshold sweep ({label}):")
            print(f"  {'Threshold':>10} | {'Precision':>9} | {'Recall':>9} | "
                  f"{'F1':>9} | {'Detected':>8}")
            print(f"  {'-'*10}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*8}")
            for r in sweep:
                marker = " ◀ best F1" if r["threshold"] == best_thr else ""
                det = int((proba_arr >= r["threshold"]).sum())
                print(f"  {r['threshold']:>10.3f} | {r['precision']:>9.4f} | "
                      f"{r['recall']:>9.4f} | {r['f1']:>9.4f} | {det:>8}{marker}")

            report[f"{label.lower().split()[0]}_sweep"] = sweep
            report[f"{label.lower().split()[0]}_best_threshold"] = best_thr

    if hailo_ok:
        _run_metrics_and_sweep(proba, "Hailo")

    # ONNX comparison (or ONNX-only fallback when Hailo unavailable)
    if args.onnx:
        onnx_path = Path(args.onnx)
        if not onnx_path.exists():
            print(f"\n  [WARN] ONNX file not found: {onnx_path}")
        else:
            mode = "comparison" if hailo_ok else "ONNX-only mode (no Hailo hardware)"
            print(f"\n  ONNX {mode} ({onnx_path.name})...")
            try:
                onnx_out = run_onnx_inference(onnx_path, Xs)
                print(f"    ONNX raw: min={onnx_out.min():.4f} "
                      f"max={onnx_out.max():.4f} mean={onnx_out.mean():.4f}")
                _run_metrics_and_sweep(onnx_out, "ONNX")

                if hailo_ok:
                    hailo_pred = (proba >= args.threshold).astype(int)
                    onnx_pred = (onnx_out >= 0.5).astype(int)
                    agree = int((hailo_pred == onnx_pred).sum())
                    agree_pct = 100.0 * agree / max(len(hailo_pred), 1)
                    print(f"\n  Hailo/ONNX agreement: {agree}/{len(hailo_pred)} ({agree_pct:.1f}%)")
                    report["hailo_onnx_agreement"] = agree
                    report["hailo_onnx_agreement_pct"] = round(agree_pct, 2)
            except Exception as e:
                print(f"  [WARN] ONNX inference failed: {e}")
    elif not hailo_ok:
        print(f"\n  [INFO] No Hailo hardware and no --onnx provided.")
        print(f"         Pass --onnx ids_mlp_binary.onnx to validate on this machine.")

    # ── Save report ───────────────────────────────────────────────────────────
    if args.out:
        out_path = Path(args.out)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n  Report saved to: {out_path}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
