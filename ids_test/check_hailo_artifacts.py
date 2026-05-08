#!/usr/bin/env python3
"""
check_hailo_artifacts.py

Pre-flight check for Hailo deployment artifacts.
Run this on the Raspberry Pi before starting ips_hailo.py or validate_hailo.py
to confirm every artifact is readable, internally consistent, and that the
Hailo chip can actually run the model.

Checks performed:
  1. All three files exist
  2. scaler_params.npz has the required keys and shapes
  3. Feature names file loads correctly and count matches scaler
  4. HEF file loads, I/O shapes are correct
  5. Synthetic all-zeros batch runs through Hailo without error

Usage:
  python3 check_hailo_artifacts.py \\
      --hef  ids_mlp_binary_logits.hef \\
      --scaler scaler_params.npz \\
      --features feature_names.txt
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
INFO = "      "


def _section(title: str):
    print(f"\n── {title} {'─' * max(0, 50 - len(title))}")


# ── 1. Scaler ────────────────────────────────────────────────────────────────

def check_scaler(npz_path: Path) -> dict:
    _section(f"Scaler: {npz_path.name}")
    required = {"mean", "scale", "with_mean", "with_std", "n_features_in"}

    try:
        data = np.load(npz_path)
    except Exception as e:
        print(f"{FAIL} Cannot load file: {e}")
        return {}

    present = set(data.keys())
    missing = required - present
    if missing:
        print(f"{FAIL} Missing keys: {missing}")
        return {}
    print(f"{PASS} All required keys present: {sorted(present)}")

    sp = {
        "mean": data["mean"].astype(np.float32),
        "scale": data["scale"].astype(np.float32),
        "with_mean": bool(int(data["with_mean"][0])),
        "with_std": bool(int(data["with_std"][0])),
        "n_features_in": int(data["n_features_in"][0]),
    }
    print(f"{INFO} n_features_in = {sp['n_features_in']}")
    print(f"{INFO} mean shape    = {sp['mean'].shape}")
    print(f"{INFO} scale shape   = {sp['scale'].shape}")
    print(f"{INFO} with_mean     = {sp['with_mean']}")
    print(f"{INFO} with_std      = {sp['with_std']}")

    if sp["mean"].shape[0] == sp["n_features_in"]:
        print(f"{PASS} mean/scale shape matches n_features_in")
    else:
        print(f"{FAIL} mean.shape[0]={sp['mean'].shape[0]} != n_features_in={sp['n_features_in']}")

    zero_scales = int((sp["scale"] == 0).sum())
    if zero_scales:
        print(f"{WARN} {zero_scales} zero value(s) in scale "
              f"(treated as 1.0 at runtime — this is normal for constant features)")
    else:
        print(f"{PASS} No zero values in scale")

    return sp


# ── 2. Feature names ─────────────────────────────────────────────────────────

def check_features(feats_path: Path, expected_n: int) -> list:
    _section(f"Features: {feats_path.name}")
    suf = feats_path.suffix.lower()

    try:
        if suf in (".txt", ".lst"):
            feats = [l.strip().lower() for l in
                     feats_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        elif suf == ".npy":
            arr = np.load(feats_path, allow_pickle=True)
            feats = [str(x).strip().lower() for x in arr.tolist()]
        else:
            print(f"{FAIL} Unsupported format '{suf}'. Use .txt or .npy")
            return []
    except Exception as e:
        print(f"{FAIL} Cannot load: {e}")
        return []

    print(f"{PASS} Loaded {len(feats)} feature names")
    print(f"{INFO} First 5: {feats[:5]}")
    print(f"{INFO} Last  5: {feats[-5:]}")

    if len(feats) == expected_n:
        print(f"{PASS} Feature count ({len(feats)}) matches scaler n_features_in ({expected_n})")
    else:
        print(f"{FAIL} Feature count {len(feats)} != scaler n_features_in {expected_n}")

    dupes = len(feats) - len(set(feats))
    if dupes:
        print(f"{WARN} {dupes} duplicate feature name(s) detected")
    else:
        print(f"{PASS} No duplicate feature names")

    return feats


# ── 3. HEF ──────────────────────────────────────────────────────────────────

def check_hef(hef_path: Path, expected_features: int) -> bool:
    _section(f"HEF: {hef_path.name}")
    size_mb = hef_path.stat().st_size / 1024 / 1024
    print(f"{INFO} File size: {size_mb:.2f} MB")

    try:
        from hailo_platform import (
            HEF, VDevice, ConfigureParams, HailoStreamInterface,
            InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams,
        )
    except Exception as e:
        print(f"{WARN} hailo_platform not available: {e}")
        print(f"{INFO} HEF file exists and is the right size — full check requires Raspberry Pi.")
        return None

    # Load HEF
    try:
        hef = HEF(str(hef_path))
        print(f"{PASS} HEF file parsed successfully")
    except Exception as e:
        print(f"{FAIL} Cannot parse HEF: {e}")
        return False

    # Inspect I/O stream info
    try:
        in_infos = hef.get_input_vstream_infos()
        out_infos = hef.get_output_vstream_infos()
        if not in_infos:
            print(f"{FAIL} HEF has no input vstreams")
            return False
        if not out_infos:
            print(f"{FAIL} HEF has no output vstreams")
            return False

        in_info = in_infos[0]
        out_info = out_infos[0]
        n_feats = int(np.prod(tuple(in_info.shape))) if in_info.shape else 1

        print(f"{INFO} Input  stream: name='{in_info.name}' shape={tuple(in_info.shape)}"
              f" → {n_feats} features")
        print(f"{INFO} Output stream: name='{out_info.name}' shape={tuple(out_info.shape)}")

        if n_feats == expected_features:
            print(f"{PASS} HEF input features ({n_feats}) match expected ({expected_features})")
        else:
            print(f"{FAIL} HEF input features={n_feats} != expected {expected_features}")
    except Exception as e:
        print(f"{WARN} Could not inspect stream info: {e}")
        in_info = None
        out_info = None

    # Configure on VDevice
    _section("Synthetic inference test (4 all-zero rows)")
    try:
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
        in_shape = tuple(in_info.shape)
        n_feats = int(np.prod(in_shape)) if in_shape else 1

        batch = 4
        x_syn = np.zeros((batch, n_feats), dtype=np.float32)
        if len(in_shape) > 1:
            x_syn = x_syn.reshape((batch,) + in_shape)
        x_syn = np.ascontiguousarray(x_syn)

        with ng.activate(ng_params):
            with InferVStreams(ng, in_params, out_params) as pipe:
                results = pipe.infer({in_info.name: x_syn})

        raw = np.asarray(results[out_info.name]).ravel()
        print(f"{PASS} Synthetic inference completed — output shape {raw.shape}")
        print(f"{INFO} Raw outputs (first 4): {raw[:4]}")

        if raw.min() < 0 or raw.max() > 1:
            print(f"{WARN} Output is outside [0,1] → model outputs LOGITS")
            print(f"{INFO} Apply sigmoid at runtime (ips_hailo.py does this automatically)")
        else:
            print(f"{PASS} Output is in [0,1] → model outputs PROBABILITIES")
            print(f"{INFO} Use --no-sigmoid if running validate_hailo.py")

        return True

    except Exception as e:
        print(f"{FAIL} Synthetic inference failed: {e}")
        return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Pre-flight check for Hailo deployment artifacts (run on Raspberry Pi).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--hef", required=True, help="HEF model file")
    ap.add_argument("--scaler", required=True, help="scaler_params.npz")
    ap.add_argument("--features", required=True, help="Feature names file (.txt or .npy)")
    args = ap.parse_args()

    hef_path = Path(args.hef)
    scaler_path = Path(args.scaler)
    feats_path = Path(args.features)

    print(f"\n{'='*55}")
    print("  HAILO ARTIFACT PRE-FLIGHT CHECK")
    print(f"{'='*55}")

    # File existence
    missing = []
    for label, p in [("HEF", hef_path), ("Scaler", scaler_path), ("Features", feats_path)]:
        if p.exists():
            print(f"{PASS} Found {label}: {p}")
        else:
            print(f"{FAIL} Missing {label}: {p}")
            missing.append(label)

    if missing:
        sys.exit(f"\nAborting: {len(missing)} file(s) not found.")

    sp = check_scaler(scaler_path)
    if not sp:
        sys.exit("\nAborting: scaler check failed.")

    feats = check_features(feats_path, sp["n_features_in"])
    if not feats:
        sys.exit("\nAborting: features check failed.")

    ok = check_hef(hef_path, sp["n_features_in"])

    print(f"\n{'='*55}")
    if ok is None:
        print("  Scaler + features OK. HEF check skipped (no Hailo hardware).")
        print("  Re-run on Raspberry Pi for the full check.")
    elif ok:
        print("  All checks passed. Ready to run ips_hailo.py.")
    else:
        print("  Some checks FAILED. Resolve the issues above before running.")
    print(f"{'='*55}\n")

    sys.exit(0 if ok is not False else 1)


if __name__ == "__main__":
    main()
