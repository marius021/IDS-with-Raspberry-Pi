#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import json
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from bench_timing import StageTimer, maybe_writer
import onnxruntime as ort

DEFAULT_BASE = Path.home() / "ids"
DEFAULT_INPUT = DEFAULT_BASE / "live_sample.csv"
DEFAULT_MODEL = DEFAULT_BASE / "ids_mlp_binary.onnx"
DEFAULT_SCALER = DEFAULT_BASE / "scaler.joblib"
DEFAULT_FEATS = DEFAULT_BASE / "feature_names.npy"
DEFAULT_ALERT_LOG = DEFAULT_BASE / "alerts.log"
DEFAULT_ACTION_LOG = DEFAULT_BASE / "actions.log"

POLL_SEC = 3
THRESHOLD = 0.001
BATCH_SIZE = 1024

SRC_IP_CANDIDATES = [
    "src ip", "src_ip", "source ip", "source_ip", "ip src", "ip_src"
]

DEFAULT_WHITELIST = {
    "127.0.0.1",
    "0.0.0.0",
    "::1",
}


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
        .str.replace("﻿", "", regex=False)
        .str.strip()
        .str.lower()
    )
    return df


def detect_src_ip_column(df: pd.DataFrame):
    cols = list(df.columns)
    for cand in SRC_IP_CANDIDATES:
        if cand in cols:
            return cand
    return None


def build_feature_matrix(df: pd.DataFrame, scaler, feats_path: Path):
    df = normalize_columns(df)
    df_num = df.select_dtypes(include=[np.number]).copy()
    df_num = clean_numeric(df_num)

    wanted = np.load(feats_path, allow_pickle=True).tolist()
    wanted = [str(c).strip().lower() for c in wanted]

    missing = [c for c in wanted if c not in df_num.columns]
    if missing:
        raise RuntimeError(
            "Columns missing compared to the training set: " + ", ".join(missing[:20])
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


def load_whitelist(path):
    whitelist = set(DEFAULT_WHITELIST)
    if path and path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            ip = line.strip()
            if ip and not ip.startswith("#"):
                whitelist.add(ip)
    return whitelist


def is_ip_blocked(ip: str) -> bool:
    try:
        result = subprocess.run(
            ["sudo", "iptables", "-C", "INPUT", "-s", ip, "-j", "DROP"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def block_ip(ip: str, dry_run: bool = True):
    cmd = ["sudo", "iptables", "-A", "INPUT", "-s", ip, "-j", "DROP"]
    if dry_run:
        return {"status": "dry-run", "cmd": " ".join(cmd)}

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return {"status": "blocked", "cmd": " ".join(cmd)}
    return {
        "status": "error",
        "cmd": " ".join(cmd),
        "stderr": result.stderr.strip(),
        "stdout": result.stdout.strip(),
        "returncode": result.returncode,
    }


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
        print(f"[ALERT] {attacks} ATTACK event(s) (top prob: {top_prob:.3f})")


def append_actions(rows_df, prob, pred, src_ip_col, action_log: Path,
                   whitelist, dry_run: bool, seen_cache: set, debug: bool = False):
    ts = int(time.time())
    action_log.parent.mkdir(parents=True, exist_ok=True)

    if src_ip_col is None:
        if int(pred.sum()) > 0:
            print("[WARN] Attack detected, but no src ip column found. Cannot apply blocking.")
        return

    with open(action_log, "a", encoding="utf-8") as f:
        for i, is_attack in enumerate(pred):
            if not is_attack:
                continue

            raw_ip = str(rows_df.iloc[i][src_ip_col]).strip()
            rec = {
                "ts": ts,
                "row_index": int(rows_df.index[i]),
                "prob": float(prob[i]),
                "src_ip": raw_ip,
            }

            if not raw_ip or raw_ip.lower() == "nan":
                rec["action"] = "skip"
                rec["reason"] = "invalid_src_ip"
                f.write(json.dumps(rec) + "\n")
                continue

            if raw_ip in whitelist:
                rec["action"] = "skip"
                rec["reason"] = "whitelisted"
                f.write(json.dumps(rec) + "\n")
                if debug:
                    print(f"[DEBUG] SKIP whitelist: {raw_ip}")
                continue

            if raw_ip in seen_cache:
                rec["action"] = "skip"
                rec["reason"] = "already_seen_this_run"
                f.write(json.dumps(rec) + "\n")
                if debug:
                    print(f"[DEBUG] SKIP already seen in current session: {raw_ip}")
                continue

            if is_ip_blocked(raw_ip):
                rec["action"] = "skip"
                rec["reason"] = "already_blocked"
                f.write(json.dumps(rec) + "\n")
                seen_cache.add(raw_ip)
                if debug:
                    print(f"[DEBUG] SKIP already blocked: {raw_ip}")
                continue

            result = block_ip(raw_ip, dry_run=dry_run)
            rec["action"] = result["status"]
            rec["cmd"] = result["cmd"]
            if "stderr" in result:
                rec["stderr"] = result["stderr"]
            if "stdout" in result:
                rec["stdout"] = result["stdout"]
            if "returncode" in result:
                rec["returncode"] = result["returncode"]

            f.write(json.dumps(rec) + "\n")
            seen_cache.add(raw_ip)

            if result["status"] == "blocked":
                print(f"[IPS] BLOCKED {raw_ip} (prob={float(prob[i]):.3f})")
            elif result["status"] == "dry-run":
                print(f"[IPS] DRY-RUN would block {raw_ip} (prob={float(prob[i]):.3f})")
            else:
                print(f"[IPS] ERROR blocking {raw_ip}: {result.get('stderr', 'unknown error')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT), help="Source CSV (single file that grows)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="ONNX model (binary)")
    ap.add_argument("--scaler", default=str(DEFAULT_SCALER), help="scaler.joblib")
    ap.add_argument("--features", default=str(DEFAULT_FEATS), help="feature_names.npy")
    ap.add_argument("--alert-log", default=str(DEFAULT_ALERT_LOG), help="alert log file")
    ap.add_argument("--action-log", default=str(DEFAULT_ACTION_LOG), help="IPS action log file")
    ap.add_argument("--whitelist", default="", help="text file with whitelisted IPs, one per line")
    ap.add_argument("--poll", type=int, default=POLL_SEC, help="poll interval (sec)")
    ap.add_argument("--batch", type=int, default=BATCH_SIZE, help="batch size")
    ap.add_argument("--threshold", type=float, default=THRESHOLD, help="binary threshold for ATTACK")
    ap.add_argument("--dry-run", action="store_true", help="do not apply iptables, only log what would be blocked")
    ap.add_argument("--debug", action="store_true", help="show debug messages")
    args = ap.parse_args()

    input_csv = Path(args.input)
    model_p = Path(args.model)
    scaler_p = Path(args.scaler)
    feats_p = Path(args.features)
    alert_log = Path(args.alert_log)
    action_log = Path(args.action_log)
    whitelist_path = Path(args.whitelist) if args.whitelist else None

    if not model_p.exists():
        raise FileNotFoundError(f"Model does not exist: {model_p}")
    if not scaler_p.exists():
        raise FileNotFoundError(f"Scaler does not exist: {scaler_p}")
    if not feats_p.exists():
        raise FileNotFoundError(f"Feature list does not exist: {feats_p}")

    whitelist = load_whitelist(whitelist_path)
    scaler = joblib.load(scaler_p)
    sess = ort.InferenceSession(str(model_p), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    last_seen = 0
    seen_cache = set()

    print(
        f"[INFO] Started. Monitoring: {input_csv} | poll={args.poll}s | "
        f"batch={args.batch} | threshold={args.threshold} | dry_run={args.dry_run}"
    )
    print(f"[INFO] Alert log: {alert_log}")
    print(f"[INFO] Action log: {action_log}")
    print(f"[INFO] Whitelist size: {len(whitelist)}")

    bench = maybe_writer(default_path="timing_cpu.csv", default_variant="cpu")
    timer = StageTimer()
    batch_idx = 0

    while True:
        if input_csv.exists():
            try:
                df_all = pd.read_csv(input_csv, low_memory=False)
                df_all = normalize_columns(df_all)
                n = len(df_all)

                if args.debug:
                    print(f"[DEBUG] total rows in file: {n} | last_seen: {last_seen}")

                if n < last_seen:
                    print(f"[INFO] File recreated or truncated. Resetting last_seen: {last_seen} -> 0")
                    last_seen = 0
                    seen_cache.clear()

                if n > last_seen:
                    df_new = df_all.iloc[last_seen:n].copy()

                    if args.debug:
                        print(f"[DEBUG] new rows: {len(df_new)}")

                    src_ip_col = detect_src_ip_column(df_new)
                    if args.debug:
                        print(f"[DEBUG] src_ip_col: {src_ip_col}")

                    for start in range(0, len(df_new), args.batch):
                        chunk = df_new.iloc[start:start + args.batch].copy()

                        if args.debug:
                            print(f"[DEBUG] processing chunk {start}:{start + len(chunk)}")

                        timer.reset()

                        timer.start("preprocess")
                        Xs = build_feature_matrix(chunk, scaler, feats_p)
                        timer.stop("preprocess")

                        timer.start("inference")
                        prob, pred = run_batch(sess, input_name, Xs, args.threshold)
                        timer.stop("inference")

                        timer.start("postprocess")
                        # sigmoid + threshold are already in run_batch; just marking here
                        n_attacks = int(pred.sum())
                        timer.stop("postprocess")

                        if args.debug:
                            print(
                                f"[DEBUG] batch rows={len(chunk)} | "
                                f"max_prob={float(prob.max()):.6f} | attacks={n_attacks}"
                            )

                        timer.start("log")
                        append_alerts(chunk, prob, pred, alert_log)
                        append_actions(chunk, prob, pred, src_ip_col, action_log,
                                       whitelist, args.dry_run, seen_cache, args.debug)
                        timer.stop("log")

                        if bench:
                            bench.write_batch(batch_idx, len(chunk), timer)
                            batch_idx += 1

                    last_seen = n
                else:
                    print("[DEBUG] No new rows to process.")
            except Exception as e:
                print(f"[WARN] Error during processing: {e}")
        else:
            print(f"[INFO] Waiting for input file: {input_csv}")

        time.sleep(args.poll)


if __name__ == "__main__":
    main()
