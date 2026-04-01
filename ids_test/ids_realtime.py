#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import onnxruntime as ort

# ---------- Config implicită (poți suprascrie din CLI) ----------
DEFAULT_BASE = Path.home() / "ids"
DEFAULT_INPUT = DEFAULT_BASE / "sample.csv"
DEFAULT_MODEL = DEFAULT_BASE / "ids_mlp_binary.onnx"
DEFAULT_SCALER = DEFAULT_BASE / "scaler.joblib"
DEFAULT_FEATS = DEFAULT_BASE / "feature_names.npy"
DEFAULT_ALERT_LOG = DEFAULT_BASE / "alerts.log"

POLL_SEC = 3
THRESHOLD = 0.01
BATCH_SIZE = 1024


# ---------- Utilitare ----------
def clean_numeric(df_num: pd.DataFrame) -> pd.DataFrame:
    df_num = df_num.replace([np.inf, -np.inf], np.nan)
    df_num = df_num.clip(lower=-1e12, upper=1e12)
    med = df_num.median(numeric_only=True)
    df_num = df_num.fillna(med).fillna(0)
    return df_num


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
    )
    return df


def build_feature_matrix(df: pd.DataFrame, scaler, feats_path: Path):
    df = normalize_columns(df)
    df_num = df.select_dtypes(include=[np.number]).copy()
    df_num = clean_numeric(df_num)

    wanted = np.load(feats_path, allow_pickle=True).tolist()
    wanted = [str(c).strip().lower() for c in wanted]

    missing = [c for c in wanted if c not in df_num.columns]
    if missing:
        raise RuntimeError(
            "Lipsesc coloane față de setul de antrenare: " + ", ".join(missing[:20])
        )

    df_num = df_num[wanted]
    X = df_num.values.astype(np.float32)
    Xs = scaler.transform(X).astype(np.float32)
    return Xs


def run_batch(sess, input_name, Xs, threshold: float):
    outs = sess.run(None, {input_name: Xs})
    proba = np.asarray(outs[0]).ravel().astype(float)
    pred = (proba >= threshold).astype(int)
    return proba, pred


def append_alerts(rows_df: pd.DataFrame, prob, pred, alert_log: Path):
    ts = int(time.time())
    alert_log.parent.mkdir(parents=True, exist_ok=True)

    with open(alert_log, "a", encoding="utf-8") as f:
        for i, is_attack in enumerate(pred):
            if is_attack:
                rec = {
                    "ts": ts,
                    "prob": float(prob[i]),
                    "row_index": int(rows_df.index[i]),
                }
                f.write(json.dumps(rec) + "\n")

    attacks = int(pred.sum())
    if attacks:
        top_prob = float(np.max(prob[pred == 1]))
        print(f"[ALERT] {attacks} eveniment(e) ATTACK (top prob: {top_prob:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT), help="CSV sursă (fișier unic care crește)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="Model ONNX (binary)")
    ap.add_argument("--scaler", default=str(DEFAULT_SCALER), help="scaler.joblib")
    ap.add_argument("--features", default=str(DEFAULT_FEATS), help="feature_names.npy")
    ap.add_argument("--alert-log", default=str(DEFAULT_ALERT_LOG), help="fișierul de log pentru alerte")
    ap.add_argument("--poll", type=int, default=POLL_SEC, help="interval de poll (sec)")
    ap.add_argument("--batch", type=int, default=BATCH_SIZE, help="mărimea batch-ului")
    ap.add_argument("--threshold", type=float, default=THRESHOLD, help="prag binary pentru ATTACK")
    ap.add_argument("--debug", action="store_true", help="afișează mesaje de debug")
    args = ap.parse_args()

    input_csv = Path(args.input)
    model_p = Path(args.model)
    scaler_p = Path(args.scaler)
    feats_p = Path(args.features)
    alert_log = Path(args.alert_log)

    if not model_p.exists():
        raise FileNotFoundError(f"Modelul nu există: {model_p}")
    if not scaler_p.exists():
        raise FileNotFoundError(f"Scalerul nu există: {scaler_p}")
    if not feats_p.exists():
        raise FileNotFoundError(f"Feature list nu există: {feats_p}")

    scaler = joblib.load(scaler_p)
    sess = ort.InferenceSession(str(model_p), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    last_seen = 0
    print(
        f"[INFO] Pornit. Monitorizez: {input_csv} | poll={args.poll}s | "
        f"batch={args.batch} | threshold={args.threshold}"
    )
    print(f"[INFO] Log alerte: {alert_log}")

    while True:
        if input_csv.exists():
            try:
                df_all = pd.read_csv(input_csv, low_memory=False)
                n = len(df_all)

                if args.debug:
                    print(f"[DEBUG] total rows in file: {n} | last_seen: {last_seen}")
                
                if n < last_seen:
                    print(f"[INFO] Fișier recreat sau trunchiat. Reset last_seen: {last_seen} -> 0")
                    last_seen = 0

                if n > last_seen:
                    df_new = df_all.iloc[last_seen:n]

                    if args.debug:
                        print(f"[DEBUG] linii noi: {len(df_new)}")

                    for start in range(0, len(df_new), args.batch):
                        chunk = df_new.iloc[start:start + args.batch]

                        if args.debug:
                            print(f"[DEBUG] procesez chunk {start}:{start + len(chunk)}")

                        Xs = build_feature_matrix(chunk, scaler, feats_p)
                        prob, pred = run_batch(sess, input_name, Xs, args.threshold)

                        if args.debug:
                            print(
                                f"[DEBUG] batch rows={len(chunk)} | "
                                f"max_prob={float(prob.max()):.6f} | attacks={int(pred.sum())}"
                            )

                        append_alerts(chunk, prob, pred, alert_log)

                    last_seen = n
                else:
                    print("[DEBUG] Nicio linie noua de procesat.")
            except Exception as e:
                print(f"[WARN] Eroare la procesare: {e}")
        else:
            print(f"[INFO] Aștept fișierul de intrare: {input_csv}")

        time.sleep(args.poll)


if __name__ == "__main__":
    main()
