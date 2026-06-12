#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ips_hailo.py

IPS + IDS with Hailo HEF, without scikit-learn at runtime.

Functions:
- monitors a CSV that grows over time
- loads scaler_params.npz
- loads the list of features from txt/json/csv/npy
- performs numeric preprocessing
- runs inference on Hailo
- writes alerts.log and actions.log
- can block IPs with iptables
- supports --dry-run for safe testing
"""

import time
import json
import argparse
import subprocess
from pathlib import Path
from typing import Optional, List
from bench_timing import StageTimer, maybe_writer

import numpy as np
import pandas as pd

# -------- Hailo imports --------
try:
    from hailo_platform import (
        HEF,
        VDevice,
        ConfigureParams,
        HailoStreamInterface,
        InputVStreamParams,
        OutputVStreamParams,
        FormatType,
        InferVStreams,
    )
except Exception as e:
    raise SystemExit(
        "Could not import Hailo Python bindings (`hailo_platform`).\n"
        f"Error: {e}\n"
        "Make sure HailoRT Python bindings are installed on the Raspberry Pi."
    )

# ---------- Default config ----------
DEFAULT_BASE = Path.home() / "Desktop" / "IDS-with-Raspberry-Pi" / "ids_test"
DEFAULT_INPUT = DEFAULT_BASE / "live_sample.csv"
DEFAULT_HEF = DEFAULT_BASE / "ids_mlp_binary.hef"
DEFAULT_SCALER = DEFAULT_BASE / "scaler_params.npz"
DEFAULT_ALERT_LOG = DEFAULT_BASE / "alerts_hailo.log"
DEFAULT_ACTION_LOG = DEFAULT_BASE / "actions_hailo.log"

POLL_SEC = 3
THRESHOLD = 0.5
BATCH_SIZE = 1024

SRC_IP_CANDIDATES = [
    "src ip", "src_ip", "source ip", "source_ip", "ip src", "ip_src"
]

DEFAULT_WHITELIST = {
    "127.0.0.1",
    "0.0.0.0",
    "::1",
}


# ---------- Scaler loader ----------
def load_scaler_params(npz_path: Path):
    data = np.load(npz_path)
    return {
        "mean": data["mean"].astype(np.float32),
        "scale": data["scale"].astype(np.float32),
        "with_mean": bool(int(data["with_mean"][0])),
        "with_std": bool(int(data["with_std"][0])),
        "n_features_in": int(data["n_features_in"][0]),
    }


def apply_standard_scaler(X: np.ndarray, scaler_params) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    out = X.copy()

    if scaler_params["with_mean"]:
        out = out - scaler_params["mean"]

    if scaler_params["with_std"]:
        scale = scaler_params["scale"].copy()
        scale[scale == 0] = 1.0
        out = out / scale

    return np.ascontiguousarray(out, dtype=np.float32)


# ---------- Feature names loader ----------
def resolve_features_path(user_arg: Optional[str], base_dir: Path) -> Path:
    if user_arg:
        p = Path(user_arg).expanduser()
        if p.exists():
            return p
        raise FileNotFoundError(f"Feature names file does not exist: {p}")

    candidates = [
        base_dir / "feature_names.txt",
        base_dir / "feature_names.json",
        base_dir / "feature_names.csv",
        base_dir / "feature_names_safe.npy",
        base_dir / "feature_names.npy",
        base_dir / "feature_names copy.npy",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Could not find the feature names file.\n"
        "Searched for: feature_names.txt, feature_names.json, feature_names.csv, "
        "feature_names_safe.npy, feature_names.npy, feature_names copy.npy"
    )


def load_feature_names(feats_path: Path) -> List[str]:
    feats_path = Path(feats_path)
    suffix = feats_path.suffix.lower()

    if suffix in [".txt", ".lst"]:
        lines = feats_path.read_text(encoding="utf-8").splitlines()
        feats = [line.strip().lower() for line in lines if line.strip()]
        if not feats:
            raise RuntimeError(f"File {feats_path} is empty.")
        return feats

    if suffix == ".json":
        obj = json.loads(feats_path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            feats = [str(x).strip().lower() for x in obj if str(x).strip()]
            if not feats:
                raise RuntimeError(f"JSON file {feats_path} is empty.")
            return feats
        if isinstance(obj, dict) and "features" in obj and isinstance(obj["features"], list):
            feats = [str(x).strip().lower() for x in obj["features"] if str(x).strip()]
            if not feats:
                raise RuntimeError(f"JSON file {feats_path} contains no valid features.")
            return feats
        raise RuntimeError(f"JSON file {feats_path} has no valid format.")

    if suffix == ".csv":
        raw = feats_path.read_text(encoding="utf-8").splitlines()
        feats = []
        for line in raw:
            line = line.strip()
            if not line:
                continue
            # take the first value on the line
            first = line.split(",")[0].strip().strip('"').strip("'")
            if first.lower() not in {"feature", "features", "name"}:
                feats.append(first.lower())
        if not feats:
            raise RuntimeError(f"CSV file {feats_path} is empty or invalid.")
        return feats

    if suffix == ".npy":
        # First attempt without pickle
        try:
            arr = np.load(feats_path, allow_pickle=False)
            return [str(x).strip().lower() for x in arr.tolist()]
        except Exception:
            pass

        # Then attempt with pickle
        try:
            arr = np.load(feats_path, allow_pickle=True)
            return [str(x).strip().lower() for x in arr.tolist()]
        except Exception as e:
            raise RuntimeError(
                f"Could not load {feats_path}.\n"
                f"Error: {e}\n"
                "If you see something like `No module named numpy._core`, "
                "it means the .npy file was saved with a different major version of NumPy.\n"
                "The safe solution is to convert the feature list to `feature_names.txt` on your PC "
                "and run the script with that file."
            )

    raise RuntimeError(
        f"Unsupported extension for feature names: {feats_path.suffix}\n"
        "Use .txt / .json / .csv / .npy"
    )


# ---------- Hailo backend ----------
class HailoRunner:
    def __init__(self, hef_path: Path):
        self.hef_path = Path(hef_path)
        self.hef = HEF(str(self.hef_path))
        self.device = VDevice()

        configure_params = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe
        )
        configured = self.device.configure(self.hef, configure_params)

        if isinstance(configured, (list, tuple)):
            self.network_group = configured[0]
        else:
            self.network_group = configured

        self.network_group_params = self.network_group.create_params()

        try:
            self.input_vstream_params = InputVStreamParams.make_from_network_group(
                self.network_group,
                quantized=False,
                format_type=FormatType.FLOAT32
            )
            self.output_vstream_params = OutputVStreamParams.make_from_network_group(
                self.network_group,
                quantized=False,
                format_type=FormatType.FLOAT32
            )
        except Exception:
            self.input_vstream_params = InputVStreamParams.make(
                self.network_group,
                format_type=FormatType.FLOAT32
            )
            self.output_vstream_params = OutputVStreamParams.make(
                self.network_group,
                format_type=FormatType.FLOAT32
            )

        self.input_infos = self.hef.get_input_vstream_infos()
        self.output_infos = self.hef.get_output_vstream_infos()

        if not self.input_infos:
            raise RuntimeError("The HEF has no input vstreams.")
        if not self.output_infos:
            raise RuntimeError("The HEF has no output vstreams.")

        self.input_info = self.input_infos[0]
        self.output_info = self.output_infos[0]

        self.input_name = self.input_info.name
        self.output_name = self.output_info.name

        self.input_shape = tuple(self.input_info.shape)
        self.output_shape = tuple(self.output_info.shape)

        self.input_features = int(np.prod(self.input_shape)) if len(self.input_shape) > 0 else 1

    def _prepare_input(self, Xs: np.ndarray) -> np.ndarray:
        Xs = np.asarray(Xs, dtype=np.float32)
        Xs = np.ascontiguousarray(Xs)

        if Xs.ndim != 2:
            raise RuntimeError(f"Xs must have shape (batch, features), received: {Xs.shape}")

        if Xs.shape[1] != self.input_features:
            raise RuntimeError(
                f"Wrong number of features. Model expects {self.input_features}, "
                f"but batch has {Xs.shape[1]}. HEF input shape = {self.input_shape}"
            )

        if len(self.input_shape) <= 1:
            return Xs

        return np.ascontiguousarray(
            Xs.reshape((Xs.shape[0],) + self.input_shape),
            dtype=np.float32
        )

    def infer(self, Xs: np.ndarray) -> np.ndarray:
        input_tensor = self._prepare_input(Xs)

        with self.network_group.activate(self.network_group_params):
            with InferVStreams(
                self.network_group,
                self.input_vstream_params,
                self.output_vstream_params,
            ) as infer_pipeline:
                results = infer_pipeline.infer({self.input_name: input_tensor})

        out = np.asarray(results[self.output_name])

        if out.ndim == 0:
            return np.array([float(out)], dtype=np.float32)

        if out.ndim == 1:
            return out.astype(np.float32)

        out = out.reshape(out.shape[0], -1)
        return out[:, 0].astype(np.float32)


# ---------- Utilities ----------
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


def detect_src_ip_column(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    for cand in SRC_IP_CANDIDATES:
        if cand in cols:
            return cand
    return None


def build_feature_matrix(df: pd.DataFrame, scaler_params, wanted_features: List[str]) -> np.ndarray:
    df = normalize_columns(df)
    df_num = df.select_dtypes(include=[np.number]).copy()
    df_num = clean_numeric(df_num)

    missing = [c for c in wanted_features if c not in df_num.columns]
    if missing:
        raise RuntimeError(
            "Columns missing compared to the training set: " + ", ".join(missing[:20])
        )

    df_num = df_num[wanted_features]
    X = df_num.values.astype(np.float32)

    if X.shape[1] != scaler_params["n_features_in"]:
        raise RuntimeError(
            f"Scaler expects {scaler_params['n_features_in']} features, "
            f"but batch has {X.shape[1]}"
        )

    Xs = apply_standard_scaler(X, scaler_params)
    return np.ascontiguousarray(Xs, dtype=np.float32)


def sigmoid(x):
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))

def run_batch_hailo(runner, Xs, threshold):
    logits = runner.infer(Xs)
    proba = sigmoid(logits).astype(np.float32)
    pred = (proba >= threshold).astype(int)
    return proba, pred


def load_whitelist(path: Optional[Path]):
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
    ap.add_argument("--input", default=str(DEFAULT_INPUT), help="Source CSV")
    ap.add_argument("--hef", default=str(DEFAULT_HEF), help="Hailo HEF model")
    ap.add_argument("--scaler", default=str(DEFAULT_SCALER), help="scaler_params.npz")
    ap.add_argument(
        "--features",
        default="",
        help="file with feature names (.txt/.json/.csv/.npy). If missing, the script searches automatically."
    )
    ap.add_argument("--alert-log", default=str(DEFAULT_ALERT_LOG), help="alert log")
    ap.add_argument("--action-log", default=str(DEFAULT_ACTION_LOG), help="IPS action log")
    ap.add_argument("--whitelist", default="", help="text file with whitelisted IPs")
    ap.add_argument("--poll", type=int, default=POLL_SEC, help="poll interval (sec)")
    ap.add_argument("--batch", type=int, default=BATCH_SIZE, help="batch size")
    ap.add_argument("--threshold", type=float, default=THRESHOLD, help="binary threshold")
    ap.add_argument("--dry-run", action="store_true", help="do not apply iptables")
    ap.add_argument("--debug", action="store_true", help="show debug output")
    args = ap.parse_args()

    input_csv = Path(args.input).expanduser()
    hef_p = Path(args.hef).expanduser()
    scaler_p = Path(args.scaler).expanduser()
    base_dir = input_csv.parent
    feats_p = resolve_features_path(args.features, base_dir)
    alert_log = Path(args.alert_log).expanduser()
    action_log = Path(args.action_log).expanduser()
    whitelist_path = Path(args.whitelist).expanduser() if args.whitelist else None

    if not hef_p.exists():
        raise FileNotFoundError(f"HEF does not exist: {hef_p}")
    if not scaler_p.exists():
        raise FileNotFoundError(f"Scaler does not exist: {scaler_p}")
    if not feats_p.exists():
        raise FileNotFoundError(f"Feature list does not exist: {feats_p}")

    whitelist = load_whitelist(whitelist_path)
    scaler = load_scaler_params(scaler_p)
    wanted_features = load_feature_names(feats_p)
    runner = HailoRunner(hef_p)

    last_seen = 0
    seen_cache = set()

    print(
        f"[INFO] Started HAILO IPS. Monitoring: {input_csv} | poll={args.poll}s | "
        f"batch={args.batch} | threshold={args.threshold} | dry_run={args.dry_run}"
    )
    print(f"[INFO] HEF: {hef_p}")
    print(f"[INFO] Scaler params: {scaler_p}")
    print(f"[INFO] Features file: {feats_p}")
    print(f"[INFO] Number of expected features: {len(wanted_features)}")
    print(f"[INFO] Alert log: {alert_log}")
    print(f"[INFO] Action log: {action_log}")
    print(f"[INFO] Whitelist size: {len(whitelist)}")
    print(f"[INFO] Hailo input stream: {runner.input_name} | output stream: {runner.output_name}")
    print(f"[INFO] Hailo input shape: {runner.input_shape}")
    print(f"[INFO] Hailo output shape: {runner.output_shape}")

    bench = maybe_writer(default_path="timing_hailo.csv", default_variant="hailo")
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
                        Xs = build_feature_matrix(chunk, scaler, wanted_features)
                        timer.stop("preprocess")

                        timer.start("inference")
                        prob, pred = run_batch_hailo(runner, Xs, args.threshold)
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
                    if args.debug:
                        print("[DEBUG] No new rows to process.")
            except Exception as e:
                print(f"[WARN] Error during processing: {e}")
        else:
            print(f"[INFO] Waiting for input file: {input_csv}")

        time.sleep(args.poll)


if __name__ == "__main__":
    main()
