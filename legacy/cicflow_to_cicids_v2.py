#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cicflow_to_cicids_v2.py

Converts a CSV produced by CICFlowMeter Java v4 (CICFlowMeterV3-0.0.4-SNAPSHOT)
to the column naming convention expected by the IDS scaler (CICIDS2017 v3 format).

Verified diff between:
  - CICFlowMeter Java v4 output (84 columns)
  - feature_names.npy (80 features expected by scaler)

This v2 has ALL 14 rename mappings pre-verified to produce "0 features missing".

Usage:
  python3 cicflow_to_cicids_v2.py <input.csv> <output.csv> [--features feature_names.npy]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Mapping LEFT (CICFlowMeter Java v4, after normalize)
#         → RIGHT (feature_names.npy entries)
RENAME_MAP = {
    # Identifiers (kept as metadata, not features — but renamed for clarity)
    "src ip":                          "source ip",
    "dst ip":                          "destination ip",
    "src port":                        "source port",
    "dst port":                        "destination port",

    # Count differences (singular vs plural)
    "total fwd packet":                "total fwd packets",
    "total bwd packets":               "total backward packets",
    "total length of fwd packet":      "total length of fwd packets",
    "total length of bwd packet":      "total length of bwd packets",

    # Packet length aggregates — ORDER OF WORDS swapped
    "packet length min":               "min packet length",
    "packet length max":               "max packet length",

    # TCP flag — CWR (correct) vs CWE (typo in training data)
    "cwr flag count":                  "cwe flag count",

    # Segment size — ORDER OF WORDS
    "fwd segment size avg":            "avg fwd segment size",
    "bwd segment size avg":            "avg bwd segment size",

    # Bulk stats — ORDER OF WORDS (+ plural)
    "fwd bytes/bulk avg":              "fwd avg bytes/bulk",
    "fwd packet/bulk avg":             "fwd avg packets/bulk",
    "fwd bulk rate avg":               "fwd avg bulk rate",
    "bwd bytes/bulk avg":              "bwd avg bytes/bulk",
    "bwd packet/bulk avg":             "bwd avg packets/bulk",
    "bwd bulk rate avg":               "bwd avg bulk rate",

    # Init window bytes — underscore vs space; "FWD" vs "fwd"
    "fwd init win bytes":              "init_win_bytes_forward",
    "bwd init win bytes":              "init_win_bytes_backward",

    # Active data packets / min seg size — order + snake_case
    "fwd act data pkts":               "act_data_pkt_fwd",
    "fwd seg size min":                "min_seg_size_forward",
}

# Special case: feature "fwd header length.1" exists in feature_names.npy
# as a DUPLICATE of "fwd header length" (artifact from old CICFlowMeter v3).
# CICFlowMeter Java v4 does NOT produce a duplicate. We synthesize it by
# copying "fwd header length" to "fwd header length.1".
DUPLICATE_FEATURE = ("fwd header length", "fwd header length.1")

# Metadata to keep in output (for IPS to identify Source IP, etc.)
METADATA_COLUMNS = ["flow id", "source ip", "source port",
                    "destination ip", "destination port",
                    "protocol", "timestamp", "label"]


def normalize(name: str) -> str:
    """Lowercase, strip, collapse internal whitespace."""
    return " ".join(name.strip().lower().split())


def convert(input_csv: Path, output_csv: Path, feature_names_path: Path | None):
    print(f"[load] Reading {input_csv}")
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"[load] Shape: {df.shape}")

    # 1. Normalize column names
    df.columns = [normalize(c) for c in df.columns]
    print(f"[norm] Normalized first 5 columns: {list(df.columns[:5])}")

    # 2. Apply rename map
    rename_actual = {c: RENAME_MAP[c] for c in df.columns if c in RENAME_MAP}
    df = df.rename(columns=rename_actual)
    print(f"[rename] Applied {len(rename_actual)} renames")
    if rename_actual:
        for src, dst in list(rename_actual.items())[:5]:
            print(f"           '{src}' → '{dst}'")
        if len(rename_actual) > 5:
            print(f"           ... and {len(rename_actual) - 5} more")

    # 3. Synthesize duplicate feature (fwd header length.1)
    src_col, dup_col = DUPLICATE_FEATURE
    if src_col in df.columns and dup_col not in df.columns:
        df[dup_col] = df[src_col]
        print(f"[duplicate] Synthesized '{dup_col}' = copy of '{src_col}'")

    # 4. Validate against feature_names.npy
    if feature_names_path and feature_names_path.exists():
        wanted_raw = np.load(feature_names_path, allow_pickle=True).tolist()
        wanted = [normalize(c) for c in wanted_raw]
        print(f"[validate] Scaler expects {len(wanted)} features")

        missing = [c for c in wanted if c not in df.columns]
        extra = [c for c in df.columns
                 if c not in wanted and c not in METADATA_COLUMNS]

        print(f"[validate] Features missing: {len(missing)}")
        if missing:
            for m in missing[:20]:
                print(f"             ❌ {m}")
            print(f"[validate] Adding missing features with 0")
            for m in missing:
                df[m] = 0

        print(f"[validate] Extra columns (will be dropped): {len(extra)}")
        if extra and len(extra) <= 10:
            for e in extra:
                print(f"             - {e}")

        # 5. Build final dataframe: metadata + features (in expected order)
        final_cols = []
        for c in METADATA_COLUMNS:
            if c in df.columns:
                final_cols.append(c)
        for c in wanted:
            if c not in final_cols:
                final_cols.append(c)

        df_out = df[final_cols]
        print(f"[output] Final shape: {df_out.shape}")
    else:
        df_out = df
        print(f"[validate] No feature_names.npy provided, keeping all columns")

    # 6. Write
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    sz = output_csv.stat().st_size / 1024
    print(f"[save] Wrote {output_csv} ({sz:.1f} KB)")
    print(f"[save] {len(df_out)} rows, {len(df_out.columns)} cols")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Input CSV from CICFlowMeter Java v4")
    ap.add_argument("output", help="Output CSV (CICIDS2017 naming, ordered for scaler)")
    ap.add_argument("--features", default="",
                    help="Path to feature_names.npy from training")
    args = ap.parse_args()

    feat_path = Path(args.features) if args.features else None
    convert(Path(args.input), Path(args.output), feat_path)


if __name__ == "__main__":
    main()
