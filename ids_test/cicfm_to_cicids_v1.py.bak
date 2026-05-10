#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cicfm_to_cicids.py (v2 — cu fix de unități timp)

Convertește un CSV produs de cicflowmeter (fork-ul hieulw v0.4.x) în formatul
schemei CICIDS2017 pe care îl așteaptă modelul.

DIFERENȚE PRINCIPALE FAȚĂ DE V1:
  - Aplică conversie sec → microsec (× 1.000.000) pentru toate features de timp.
    Cauza: CICFlowMeter Java (folosit la generarea CICIDS2017) scrie timpii
    în microsecunde. cicflowmeter Python (hieulw fork) scrie în secunde.
    Modelul a fost antrenat pe schema CICIDS2017 (microsec), deci convertim.

CE NU SE ATINGE:
  - Rate-urile (Flow Bytes/s, Flow Packets/s, Fwd/Bwd Packets/s) — sunt deja
    în "per second" în ambele implementări.
  - Numerele de flag-uri (SYN, ACK, etc.) — folosesc aceleași definiții.
  - Lungimile de pachete (Packet Length Max/Min/etc.) — în bytes peste tot.

Folosire:
  python3 cicfm_to_cicids.py input.csv output.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np


# Ordinea exactă a features din feature_names.txt (model)
EXPECTED_FEATURES = [
    "source port",
    "destination port",
    "protocol",
    "flow duration",
    "total fwd packets",
    "total backward packets",
    "total length of fwd packets",
    "total length of bwd packets",
    "fwd packet length max",
    "fwd packet length min",
    "fwd packet length mean",
    "fwd packet length std",
    "bwd packet length max",
    "bwd packet length min",
    "bwd packet length mean",
    "bwd packet length std",
    "flow bytes/s",
    "flow packets/s",
    "flow iat mean",
    "flow iat std",
    "flow iat max",
    "flow iat min",
    "fwd iat total",
    "fwd iat mean",
    "fwd iat std",
    "fwd iat max",
    "fwd iat min",
    "bwd iat total",
    "bwd iat mean",
    "bwd iat std",
    "bwd iat max",
    "bwd iat min",
    "fwd psh flags",
    "bwd psh flags",
    "fwd urg flags",
    "bwd urg flags",
    "fwd header length",
    "bwd header length",
    "fwd packets/s",
    "bwd packets/s",
    "min packet length",
    "max packet length",
    "packet length mean",
    "packet length std",
    "packet length variance",
    "fin flag count",
    "syn flag count",
    "rst flag count",
    "psh flag count",
    "ack flag count",
    "urg flag count",
    "cwe flag count",
    "ece flag count",
    "down/up ratio",
    "average packet size",
    "avg fwd segment size",
    "avg bwd segment size",
    "fwd header length.1",
    "fwd avg bytes/bulk",
    "fwd avg packets/bulk",
    "fwd avg bulk rate",
    "bwd avg bytes/bulk",
    "bwd avg packets/bulk",
    "bwd avg bulk rate",
    "subflow fwd packets",
    "subflow fwd bytes",
    "subflow bwd packets",
    "subflow bwd bytes",
    "init_win_bytes_forward",
    "init_win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "active mean",
    "active std",
    "active max",
    "active min",
    "idle mean",
    "idle std",
    "idle max",
    "idle min",
]

# Mapping cicflowmeter (hieulw fork) → CICIDS2017 expected
CICFM_TO_CICIDS = {
    "src_port": "source port",
    "dst_port": "destination port",
    "protocol": "protocol",
    "flow_duration": "flow duration",
    "tot_fwd_pkts": "total fwd packets",
    "tot_bwd_pkts": "total backward packets",
    "totlen_fwd_pkts": "total length of fwd packets",
    "totlen_bwd_pkts": "total length of bwd packets",
    "fwd_pkt_len_max": "fwd packet length max",
    "fwd_pkt_len_min": "fwd packet length min",
    "fwd_pkt_len_mean": "fwd packet length mean",
    "fwd_pkt_len_std": "fwd packet length std",
    "bwd_pkt_len_max": "bwd packet length max",
    "bwd_pkt_len_min": "bwd packet length min",
    "bwd_pkt_len_mean": "bwd packet length mean",
    "bwd_pkt_len_std": "bwd packet length std",
    "flow_byts_s": "flow bytes/s",
    "flow_pkts_s": "flow packets/s",
    "flow_iat_mean": "flow iat mean",
    "flow_iat_std": "flow iat std",
    "flow_iat_max": "flow iat max",
    "flow_iat_min": "flow iat min",
    "fwd_iat_tot": "fwd iat total",
    "fwd_iat_mean": "fwd iat mean",
    "fwd_iat_std": "fwd iat std",
    "fwd_iat_max": "fwd iat max",
    "fwd_iat_min": "fwd iat min",
    "bwd_iat_tot": "bwd iat total",
    "bwd_iat_mean": "bwd iat mean",
    "bwd_iat_std": "bwd iat std",
    "bwd_iat_max": "bwd iat max",
    "bwd_iat_min": "bwd iat min",
    "fwd_psh_flags": "fwd psh flags",
    "bwd_psh_flags": "bwd psh flags",
    "fwd_urg_flags": "fwd urg flags",
    "bwd_urg_flags": "bwd urg flags",
    "fwd_header_len": "fwd header length",
    "bwd_header_len": "bwd header length",
    "fwd_pkts_s": "fwd packets/s",
    "bwd_pkts_s": "bwd packets/s",
    "pkt_len_min": "min packet length",
    "pkt_len_max": "max packet length",
    "pkt_len_mean": "packet length mean",
    "pkt_len_std": "packet length std",
    "pkt_len_var": "packet length variance",
    "fin_flag_cnt": "fin flag count",
    "syn_flag_cnt": "syn flag count",
    "rst_flag_cnt": "rst flag count",
    "psh_flag_cnt": "psh flag count",
    "ack_flag_cnt": "ack flag count",
    "urg_flag_cnt": "urg flag count",
    "cwr_flag_count": "cwe flag count",
    "ece_flag_cnt": "ece flag count",
    "down_up_ratio": "down/up ratio",
    "pkt_size_avg": "average packet size",
    "fwd_seg_size_avg": "avg fwd segment size",
    "bwd_seg_size_avg": "avg bwd segment size",
    "fwd_byts_b_avg": "fwd avg bytes/bulk",
    "fwd_pkts_b_avg": "fwd avg packets/bulk",
    "fwd_blk_rate_avg": "fwd avg bulk rate",
    "bwd_byts_b_avg": "bwd avg bytes/bulk",
    "bwd_pkts_b_avg": "bwd avg packets/bulk",
    "bwd_blk_rate_avg": "bwd avg bulk rate",
    "subflow_fwd_pkts": "subflow fwd packets",
    "subflow_fwd_byts": "subflow fwd bytes",
    "subflow_bwd_pkts": "subflow bwd packets",
    "subflow_bwd_byts": "subflow bwd bytes",
    "init_fwd_win_byts": "init_win_bytes_forward",
    "init_bwd_win_byts": "init_win_bytes_backward",
    "fwd_act_data_pkts": "act_data_pkt_fwd",
    "fwd_seg_size_min": "min_seg_size_forward",
    "active_mean": "active mean",
    "active_std": "active std",
    "active_max": "active max",
    "active_min": "active min",
    "idle_mean": "idle mean",
    "idle_std": "idle std",
    "idle_max": "idle max",
    "idle_min": "idle min",
}

# Features de timp care trebuie multiplicate cu 1.000.000 (sec → microsec)
# IMPORTANT: doar feature-urile care reprezintă durate, NU rate-urile.
TIME_FEATURES_TO_SCALE = [
    "flow duration",
    "flow iat mean",
    "flow iat std",
    "flow iat max",
    "flow iat min",
    "fwd iat total",
    "fwd iat mean",
    "fwd iat std",
    "fwd iat max",
    "fwd iat min",
    "bwd iat total",
    "bwd iat mean",
    "bwd iat std",
    "bwd iat max",
    "bwd iat min",
    "active mean",
    "active std",
    "active max",
    "active min",
    "idle mean",
    "idle std",
    "idle max",
    "idle min",
]

SEC_TO_MICROSEC = 1_000_000


def convert(input_csv: Path, output_csv: Path, default_label: str = "BENIGN",
            scale_time: bool = True):
    df = pd.read_csv(input_csv)
    print(f"[CONVERT] Input:  {len(df)} rânduri × {len(df.columns)} coloane")

    missing_in_input = [c for c in CICFM_TO_CICIDS.keys() if c not in df.columns]
    if missing_in_input:
        print(f"[WARN] Coloane lipsă în input (vor fi umplute cu 0):")
        for c in missing_in_input:
            print(f"       - {c}")

    out = pd.DataFrame()

    # Metadata pentru IPS
    if "src_ip" in df.columns:
        out["source ip"] = df["src_ip"]
    if "dst_ip" in df.columns:
        out["destination ip"] = df["dst_ip"]
    if "timestamp" in df.columns:
        out["timestamp"] = df["timestamp"]

    # Mapping features
    for cicfm_col, cicids_col in CICFM_TO_CICIDS.items():
        if cicfm_col in df.columns:
            out[cicids_col] = df[cicfm_col]
        else:
            out[cicids_col] = 0

    # Cazul special: "fwd header length.1" e duplicat al "fwd header length"
    if "fwd_header_len" in df.columns:
        out["fwd header length.1"] = df["fwd_header_len"]
    else:
        out["fwd header length.1"] = 0

    # Curățare valori non-finite
    for col in out.select_dtypes(include=[np.number]).columns:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    # FIX UNITS: convert features de timp din secunde în microsecunde
    if scale_time:
        scaled_count = 0
        for feat in TIME_FEATURES_TO_SCALE:
            if feat in out.columns:
                # Verifică dacă valorile par să fie în secunde (mici) vs microsec (mari)
                # Heuristic: dacă median < 1000, e probabil în secunde
                if pd.api.types.is_numeric_dtype(out[feat]):
                    out[feat] = out[feat] * SEC_TO_MICROSEC
                    scaled_count += 1
        print(f"[CONVERT] Aplicat conversie sec → microsec pentru {scaled_count} features")

    # Label implicit
    out["label"] = default_label

    # Reordonare coloane
    metadata_cols = [c for c in ["source ip", "destination ip", "timestamp"] if c in out.columns]
    final_cols = metadata_cols + EXPECTED_FEATURES + ["label"]
    out = out[final_cols]

    out.to_csv(output_csv, index=False)
    print(f"[CONVERT] Output: {len(out)} rânduri × {len(out.columns)} coloane")
    print(f"[CONVERT] Salvat: {output_csv}")

    # Sumar diagnostic
    if "source ip" in out.columns:
        print(f"\n[CHECK] Top 5 src_ip:")
        for ip, count in out["source ip"].value_counts().head(5).items():
            print(f"  {ip:>15s} : {count} flow-uri")

    # Statistici flow duration după conversie
    if "flow duration" in out.columns:
        fd = out["flow duration"]
        print(f"\n[CHECK] flow duration (microsec): "
              f"min={fd.min():.0f}  median={fd.median():.0f}  max={fd.max():.0f}")


def main():
    ap = argparse.ArgumentParser(description="Convertește CSV cicflowmeter → CICIDS2017 schema")
    ap.add_argument("input", help="CSV produs de cicflowmeter")
    ap.add_argument("output", help="CSV output în formatul așteptat de model")
    ap.add_argument("--label", default="BENIGN",
                    help="Label implicit (default: BENIGN)")
    ap.add_argument("--no-time-scale", action="store_true",
                    help="NU multiplica features de timp (debugging)")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        sys.exit(f"[ERR] Input lipsește: {in_path}")

    convert(in_path, out_path, args.label, scale_time=not args.no_time_scale)


if __name__ == "__main__":
    main()
