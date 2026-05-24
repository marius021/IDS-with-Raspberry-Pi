#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pcap_to_cicids.py

Reads a PCAP file (captured with tcpdump or wireshark) and extracts network
flows in the CICIDS2017 CSV format — same 80 features used to train the model.

Implementation:
  - Uses scapy (pure Python, no native deps) to read packets
  - Groups packets into flows by 5-tuple: (src_ip, src_port, dst_ip, dst_port, protocol)
  - Treats packets as forward/backward based on first-seen direction
  - Closes flows on FIN/RST or 120s inactivity timeout
  - Computes ALL 80 CICIDS2017 features per flow
  - Writes CSV directly compatible with ips_hailo.py (no mapper needed)

Usage:
  python3 pcap_to_cicids.py <input.pcap> <output.csv>

Example:
  python3 pcap_to_cicids.py capture.pcap flows.csv
"""
import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from scapy.all import rdpcap, IP, IPv6, TCP, UDP


# Feature order — EXACTLY as in feature_names.npy (verified)
FEATURE_NAMES = [
    "source port", "destination port", "protocol", "flow duration",
    "total fwd packets", "total backward packets",
    "total length of fwd packets", "total length of bwd packets",
    "fwd packet length max", "fwd packet length min",
    "fwd packet length mean", "fwd packet length std",
    "bwd packet length max", "bwd packet length min",
    "bwd packet length mean", "bwd packet length std",
    "flow bytes/s", "flow packets/s",
    "flow iat mean", "flow iat std", "flow iat max", "flow iat min",
    "fwd iat total", "fwd iat mean", "fwd iat std", "fwd iat max", "fwd iat min",
    "bwd iat total", "bwd iat mean", "bwd iat std", "bwd iat max", "bwd iat min",
    "fwd psh flags", "bwd psh flags", "fwd urg flags", "bwd urg flags",
    "fwd header length", "bwd header length",
    "fwd packets/s", "bwd packets/s",
    "min packet length", "max packet length",
    "packet length mean", "packet length std", "packet length variance",
    "fin flag count", "syn flag count", "rst flag count", "psh flag count",
    "ack flag count", "urg flag count", "cwe flag count", "ece flag count",
    "down/up ratio", "average packet size",
    "avg fwd segment size", "avg bwd segment size",
    "fwd header length.1",
    "fwd avg bytes/bulk", "fwd avg packets/bulk", "fwd avg bulk rate",
    "bwd avg bytes/bulk", "bwd avg packets/bulk", "bwd avg bulk rate",
    "subflow fwd packets", "subflow fwd bytes",
    "subflow bwd packets", "subflow bwd bytes",
    "init_win_bytes_forward", "init_win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward",
    "active mean", "active std", "active max", "active min",
    "idle mean", "idle std", "idle max", "idle min",
]

# Metadata kept in CSV (NOT features, but needed for IPS to identify source IP)
METADATA_COLS = ["flow id", "source ip", "destination ip", "timestamp", "label"]

# Activity threshold (microseconds) — gaps larger = idle period
ACTIVITY_TIMEOUT_US = 5_000_000   # 5 sec


def safe_mean(values):
    return statistics.mean(values) if values else 0.0


def safe_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def safe_min(values):
    return min(values) if values else 0.0


def safe_max(values):
    return max(values) if values else 0.0


def safe_div(a, b):
    return a / b if b != 0 else 0.0


class Flow:
    """Represents one network flow (5-tuple)."""

    def __init__(self, key, first_packet_time):
        self.key = key   # (src_ip, src_port, dst_ip, dst_port, proto)
        self.src_ip, self.src_port, self.dst_ip, self.dst_port, self.proto = key
        self.start_time = first_packet_time
        self.last_time = first_packet_time
        # Packet sizes
        self.fwd_lengths = []
        self.bwd_lengths = []
        # Packet timestamps (in seconds)
        self.fwd_times = []
        self.bwd_times = []
        # Header lengths
        self.fwd_header_len = 0
        self.bwd_header_len = 0
        # TCP flags
        self.flags = {"FIN": 0, "SYN": 0, "RST": 0, "PSH": 0,
                      "ACK": 0, "URG": 0, "CWR": 0, "ECE": 0}
        self.fwd_psh = 0
        self.bwd_psh = 0
        self.fwd_urg = 0
        self.bwd_urg = 0
        # Initial window sizes (TCP)
        self.init_fwd_win = -1
        self.init_bwd_win = -1
        # Data packets (TCP with payload)
        self.fwd_data_packets = 0
        # Min segment size (forward only, TCP header)
        self.fwd_min_seg_size = float("inf")

    def add_packet(self, pkt, ts, direction):
        """direction: 'fwd' or 'bwd'"""
        self.last_time = ts

        # Total length (IP packet length, includes IP+transport+payload)
        if IP in pkt:
            ip_layer = pkt[IP]
            ip_total_len = ip_layer.len
            ip_header_len = ip_layer.ihl * 4
        elif IPv6 in pkt:
            ip_layer = pkt[IPv6]
            ip_total_len = ip_layer.plen + 40  # IPv6 header is fixed 40 bytes
            ip_header_len = 40
        else:
            return

        # Transport header
        if TCP in pkt:
            tcp_layer = pkt[TCP]
            tcp_header_len = tcp_layer.dataofs * 4
            transport_header_len = tcp_header_len
            payload_len = ip_total_len - ip_header_len - tcp_header_len
            # TCP flags
            flag_value = int(tcp_layer.flags)
            if flag_value & 0x01:  self.flags["FIN"] += 1
            if flag_value & 0x02:  self.flags["SYN"] += 1
            if flag_value & 0x04:  self.flags["RST"] += 1
            if flag_value & 0x08:  self.flags["PSH"] += 1
            if flag_value & 0x10:  self.flags["ACK"] += 1
            if flag_value & 0x20:  self.flags["URG"] += 1
            if flag_value & 0x80:  self.flags["CWR"] += 1
            if flag_value & 0x40:  self.flags["ECE"] += 1
            if direction == "fwd":
                if flag_value & 0x08:  self.fwd_psh += 1
                if flag_value & 0x20:  self.fwd_urg += 1
                if self.init_fwd_win < 0:
                    self.init_fwd_win = tcp_layer.window
                if payload_len > 0:
                    self.fwd_data_packets += 1
                if tcp_header_len < self.fwd_min_seg_size:
                    self.fwd_min_seg_size = tcp_header_len
            else:
                if flag_value & 0x08:  self.bwd_psh += 1
                if flag_value & 0x20:  self.bwd_urg += 1
                if self.init_bwd_win < 0:
                    self.init_bwd_win = tcp_layer.window
        elif UDP in pkt:
            transport_header_len = 8
        else:
            transport_header_len = 0

        # Total packet length (Layer 3 + payload, as used by CICFlowMeter)
        total_len = ip_total_len

        if direction == "fwd":
            self.fwd_lengths.append(total_len)
            self.fwd_times.append(ts)
            self.fwd_header_len += transport_header_len
        else:
            self.bwd_lengths.append(total_len)
            self.bwd_times.append(ts)
            self.bwd_header_len += transport_header_len

    def is_finished(self, current_time, timeout_sec=120):
        """Flow finished if FIN seen on both sides, RST, or timeout."""
        if (current_time - self.last_time) > timeout_sec:
            return True
        return False

    def has_fin_or_rst(self):
        return self.flags["FIN"] >= 2 or self.flags["RST"] >= 1

    def compute_features(self):
        """Computes all 80 CICIDS2017 features for this flow."""
        # Convert times to relative seconds, then to microseconds for IAT
        duration_sec = self.last_time - self.start_time
        duration_us = duration_sec * 1_000_000   # CICFlowMeter uses microseconds

        # Total counts
        total_fwd_pkts = len(self.fwd_lengths)
        total_bwd_pkts = len(self.bwd_lengths)
        total_pkts = total_fwd_pkts + total_bwd_pkts
        total_fwd_bytes = sum(self.fwd_lengths)
        total_bwd_bytes = sum(self.bwd_lengths)

        # Inter-arrival times (between consecutive packets)
        def iat(times):
            return [(times[i] - times[i-1]) * 1_000_000
                    for i in range(1, len(times))]

        all_times = sorted(self.fwd_times + self.bwd_times)
        flow_iats = iat(all_times)
        fwd_iats = iat(self.fwd_times)
        bwd_iats = iat(self.bwd_times)

        # All packet lengths
        all_lengths = self.fwd_lengths + self.bwd_lengths

        # Active/Idle (split flow on gaps > ACTIVITY_TIMEOUT_US)
        active_periods, idle_periods = [], []
        if len(all_times) >= 2:
            active_start = all_times[0]
            last_t = all_times[0]
            for t in all_times[1:]:
                gap_us = (t - last_t) * 1_000_000
                if gap_us > ACTIVITY_TIMEOUT_US:
                    active_periods.append((last_t - active_start) * 1_000_000)
                    idle_periods.append(gap_us)
                    active_start = t
                last_t = t
            active_periods.append((last_t - active_start) * 1_000_000)

        # Build feature dict
        feats = {
            "source port":                     self.src_port,
            "destination port":                self.dst_port,
            "protocol":                        self.proto,
            "flow duration":                   duration_us,
            "total fwd packets":               total_fwd_pkts,
            "total backward packets":          total_bwd_pkts,
            "total length of fwd packets":     total_fwd_bytes,
            "total length of bwd packets":     total_bwd_bytes,
            "fwd packet length max":           safe_max(self.fwd_lengths),
            "fwd packet length min":           safe_min(self.fwd_lengths),
            "fwd packet length mean":          safe_mean(self.fwd_lengths),
            "fwd packet length std":           safe_std(self.fwd_lengths),
            "bwd packet length max":           safe_max(self.bwd_lengths),
            "bwd packet length min":           safe_min(self.bwd_lengths),
            "bwd packet length mean":          safe_mean(self.bwd_lengths),
            "bwd packet length std":           safe_std(self.bwd_lengths),
            "flow bytes/s":                    safe_div(total_fwd_bytes + total_bwd_bytes,
                                                       duration_sec),
            "flow packets/s":                  safe_div(total_pkts, duration_sec),
            "flow iat mean":                   safe_mean(flow_iats),
            "flow iat std":                    safe_std(flow_iats),
            "flow iat max":                    safe_max(flow_iats),
            "flow iat min":                    safe_min(flow_iats),
            "fwd iat total":                   sum(fwd_iats),
            "fwd iat mean":                    safe_mean(fwd_iats),
            "fwd iat std":                     safe_std(fwd_iats),
            "fwd iat max":                     safe_max(fwd_iats),
            "fwd iat min":                     safe_min(fwd_iats),
            "bwd iat total":                   sum(bwd_iats),
            "bwd iat mean":                    safe_mean(bwd_iats),
            "bwd iat std":                     safe_std(bwd_iats),
            "bwd iat max":                     safe_max(bwd_iats),
            "bwd iat min":                     safe_min(bwd_iats),
            "fwd psh flags":                   self.fwd_psh,
            "bwd psh flags":                   self.bwd_psh,
            "fwd urg flags":                   self.fwd_urg,
            "bwd urg flags":                   self.bwd_urg,
            "fwd header length":               self.fwd_header_len,
            "bwd header length":               self.bwd_header_len,
            "fwd packets/s":                   safe_div(total_fwd_pkts, duration_sec),
            "bwd packets/s":                   safe_div(total_bwd_pkts, duration_sec),
            "min packet length":               safe_min(all_lengths),
            "max packet length":               safe_max(all_lengths),
            "packet length mean":              safe_mean(all_lengths),
            "packet length std":               safe_std(all_lengths),
            "packet length variance":          safe_std(all_lengths) ** 2,
            "fin flag count":                  self.flags["FIN"],
            "syn flag count":                  self.flags["SYN"],
            "rst flag count":                  self.flags["RST"],
            "psh flag count":                  self.flags["PSH"],
            "ack flag count":                  self.flags["ACK"],
            "urg flag count":                  self.flags["URG"],
            "cwe flag count":                  self.flags["CWR"],
            "ece flag count":                  self.flags["ECE"],
            "down/up ratio":                   safe_div(total_bwd_pkts, total_fwd_pkts),
            "average packet size":             safe_mean(all_lengths),
            "avg fwd segment size":            safe_mean(self.fwd_lengths),
            "avg bwd segment size":            safe_mean(self.bwd_lengths),
            "fwd header length.1":             self.fwd_header_len,  # duplicate
            "fwd avg bytes/bulk":              0,
            "fwd avg packets/bulk":            0,
            "fwd avg bulk rate":               0,
            "bwd avg bytes/bulk":              0,
            "bwd avg packets/bulk":            0,
            "bwd avg bulk rate":               0,
            "subflow fwd packets":             total_fwd_pkts,
            "subflow fwd bytes":               total_fwd_bytes,
            "subflow bwd packets":             total_bwd_pkts,
            "subflow bwd bytes":               total_bwd_bytes,
            "init_win_bytes_forward":          self.init_fwd_win if self.init_fwd_win >= 0 else -1,
            "init_win_bytes_backward":         self.init_bwd_win if self.init_bwd_win >= 0 else -1,
            "act_data_pkt_fwd":                self.fwd_data_packets,
            "min_seg_size_forward":            self.fwd_min_seg_size
                                                if self.fwd_min_seg_size != float("inf") else 0,
            "active mean":                     safe_mean(active_periods),
            "active std":                      safe_std(active_periods),
            "active max":                      safe_max(active_periods),
            "active min":                      safe_min(active_periods),
            "idle mean":                       safe_mean(idle_periods),
            "idle std":                        safe_std(idle_periods),
            "idle max":                        safe_max(idle_periods),
            "idle min":                        safe_min(idle_periods),
        }
        return feats


def extract_flows(pcap_path):
    """Reads PCAP, groups into flows, returns list of Flow objects."""
    print(f"[load] Reading {pcap_path} ...", file=sys.stderr)
    packets = rdpcap(str(pcap_path))
    print(f"[load] {len(packets)} packets read", file=sys.stderr)

    flows = {}   # key → Flow

    for pkt in packets:
        if IP in pkt:
            ip_layer = pkt[IP]
            src_ip = ip_layer.src
            dst_ip = ip_layer.dst
            proto = ip_layer.proto
        elif IPv6 in pkt:
            ip_layer = pkt[IPv6]
            src_ip = ip_layer.src
            dst_ip = ip_layer.dst
            proto = ip_layer.nh
        else:
            continue

        # Extract ports
        if TCP in pkt:
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
        elif UDP in pkt:
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport
        else:
            src_port = 0
            dst_port = 0

        # Flow key — bidirectional, normalized (smaller endpoint first)
        # so that fwd/bwd are consistent
        endpoint_a = (src_ip, src_port)
        endpoint_b = (dst_ip, dst_port)
        if endpoint_a <= endpoint_b:
            key = (src_ip, src_port, dst_ip, dst_port, proto)
            direction = "fwd"
        else:
            key = (dst_ip, dst_port, src_ip, src_port, proto)
            direction = "bwd"

        ts = float(pkt.time)
        if key not in flows:
            flows[key] = Flow(key, ts)
            # First packet defines the "forward" direction
            # (override the normalization above for this specific flow)
            flows[key].src_ip = src_ip
            flows[key].src_port = src_port
            flows[key].dst_ip = dst_ip
            flows[key].dst_port = dst_port
            direction = "fwd"
        else:
            flow = flows[key]
            if (src_ip, src_port) == (flow.src_ip, flow.src_port):
                direction = "fwd"
            else:
                direction = "bwd"

        flows[key].add_packet(pkt, ts, direction)

    print(f"[extract] {len(flows)} unique flows", file=sys.stderr)
    return list(flows.values())


def write_csv(flows, output_path):
    print(f"[save] Writing {output_path}", file=sys.stderr)
    fieldnames = METADATA_COLS + FEATURE_NAMES

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for flow in flows:
            from datetime import datetime
            row = {
                "flow id": f"{flow.src_ip}-{flow.dst_ip}-{flow.src_port}-"
                           f"{flow.dst_port}-{flow.proto}",
                "source ip": flow.src_ip,
                "destination ip": flow.dst_ip,
                "timestamp": datetime.fromtimestamp(flow.start_time).strftime(
                    "%Y-%m-%d %H:%M:%S"),
                "label": "",
            }
            feats = flow.compute_features()
            row.update(feats)
            writer.writerow(row)

    print(f"[save] Done. {len(flows)} flows written.", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Input .pcap file")
    ap.add_argument("output", help="Output .csv (CICIDS2017 format)")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        sys.exit(f"PCAP not found: {in_path}")

    flows = extract_flows(in_path)
    if not flows:
        print("[ERROR] No flows extracted (pcap empty or no IP packets)",
              file=sys.stderr)
        sys.exit(2)

    write_csv(flows, out_path)
    print(f"\n[summary] {len(flows)} flows extracted from {in_path}", file=sys.stderr)
    print(f"[summary] Output: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
