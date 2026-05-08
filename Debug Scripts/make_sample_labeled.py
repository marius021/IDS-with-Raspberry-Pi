#!/usr/bin/env python3
"""
make_sample_labeled.py

Builds a balanced BENIGN/ATTACK sample from the CICIDS2017 CSV files.
Output: ids_test/sample_labeled.csv  (usable by validate_hailo.py)

Usage:
  python3 "make_sample_labeled.py" \
      --data-dir "/home/dell/Desktop/Dataset CICIDS2017/csv/MachineLearningCSV/MachineLearningCVE" \
      --out ../ids_test/sample_labeled.csv \
      --per-file 300
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False, encoding="latin1",
                       engine="python", on_bad_lines="skip")


def find_label_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "label" in c.lower():
            return c
    raise KeyError(f"No label column in: {list(df.columns)[:8]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="Folder with CICIDS2017 *_ISCX.csv files")
    ap.add_argument("--out", default="../ids_test/sample_labeled.csv",
                    help="Output CSV path")
    ap.add_argument("--per-file", type=int, default=300,
                    help="Max attack rows to take per file (benign matched to same count)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    data_dir = Path(args.data_dir)
    out_path = Path(args.out)

    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    print(f"Found {len(csvs)} CSV file(s) in {data_dir}\n")

    attack_parts = []
    benign_parts = []

    for f in csvs:
        print(f"  Reading {f.name} ...", end=" ", flush=True)
        df = load_csv(f)
        df.columns = df.columns.str.strip()

        try:
            label_col = find_label_col(df)
        except KeyError as e:
            print(f"SKIP ({e})")
            continue

        df["Label"] = df[label_col].astype(str).str.strip()
        is_valid = df["Label"].str.upper().isin({"NAN", ""}) == False  # drop true NaN rows
        df = df[is_valid]
        is_attack = df["Label"].str.upper() != "BENIGN"

        attacks = df[is_attack]
        benign = df[~is_attack]

        n_att = min(len(attacks), args.per_file)
        n_ben = min(len(benign), n_att)

        if n_att == 0:
            print(f"0 attacks — skipping")
            continue

        att_sample = attacks.sample(n=n_att, random_state=int(rng.integers(1e6)))
        ben_sample = benign.sample(n=n_ben, random_state=int(rng.integers(1e6)))

        attack_parts.append(att_sample)
        benign_parts.append(ben_sample)

        attack_types = attacks["Label"].value_counts().to_dict()
        print(f"{n_att} attacks {attack_types}  |  {n_ben} benign")

    if not attack_parts:
        raise RuntimeError("No attack rows found across all files.")

    combined = pd.concat(attack_parts + benign_parts, ignore_index=True)
    combined = combined.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    label_dist = combined["Label"].value_counts()
    n_att_total = int((combined["Label"].str.upper() != "BENIGN").sum())
    n_ben_total = len(combined) - n_att_total

    print(f"\nFinal sample: {len(combined)} rows")
    print(f"  ATTACK : {n_att_total}")
    print(f"  BENIGN : {n_ben_total}")
    print(f"  Attack types: {label_dist[label_dist.index.str.upper() != 'BENIGN'].to_dict()}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
