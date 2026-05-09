#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resource_sampler.py

Rulează în paralel cu IPS-ul pentru a măsura resursele Pi-ului.
Eșantionează la fiecare N secunde:
  - %CPU total
  - %CPU per core
  - %CPU procesul țintă
  - RSS (RAM) procesul țintă
  - Temperatura CPU

Folosire:
  python resource_sampler.py --pid <PID_IPS> --out resources_cpu.csv --interval 1 --duration 120

Sau, dacă vrei să atașezi după nume:
  python resource_sampler.py --proc-name ips_realtime_v2.py --out resources_cpu.csv

Ctrl-C oprește elegant.
"""

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

try:
    import psutil
except ImportError:
    sys.exit("psutil lipsește. Instalează cu: pip install psutil")


def find_pid_by_name(needle: str):
    """Caută primul proces al cărui cmdline conține needle."""
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = " ".join(p.info["cmdline"] or [])
            if needle in cmdline and p.info["pid"] != os.getpid():
                return p.info["pid"]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def get_cpu_temp_celsius():
    """Citește temperatura CPU pe Raspberry Pi via vcgencmd sau /sys."""
    # Try vcgencmd first
    try:
        result = subprocess.run(
            ["vcgencmd", "measure_temp"],
            capture_output=True, text=True, timeout=1
        )
        if result.returncode == 0:
            # Output: temp=52.0'C
            t = result.stdout.strip().replace("temp=", "").replace("'C", "")
            return float(t)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    # Fallback to thermal zone
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return float(f.read().strip()) / 1000.0
    except (FileNotFoundError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, help="PID-ul procesului de monitorizat")
    ap.add_argument("--proc-name", help="Numele/cmdline-ul procesului (dacă nu ai PID)")
    ap.add_argument("--out", required=True, help="CSV-ul de output")
    ap.add_argument("--variant", default="unknown", help="Label: cpu / hailo / etc")
    ap.add_argument("--interval", type=float, default=1.0, help="Interval (sec)")
    ap.add_argument("--duration", type=float, default=0,
                    help="Durată totală (sec). 0 = până la Ctrl-C")
    args = ap.parse_args()

    # Rezolvă PID
    pid = args.pid
    if pid is None and args.proc_name:
        pid = find_pid_by_name(args.proc_name)
        if pid is None:
            sys.exit(f"Nu am găsit niciun proces cu '{args.proc_name}' în cmdline.")

    if pid is None:
        sys.exit("Trebuie să specifici --pid sau --proc-name.")

    try:
        target = psutil.Process(pid)
    except psutil.NoSuchProcess:
        sys.exit(f"Procesul {pid} nu există.")

    n_cores = psutil.cpu_count()
    print(f"[SAMPLER] PID={pid} cmdline=\"{' '.join(target.cmdline()[:3])}...\"")
    print(f"[SAMPLER] Cores: {n_cores}, interval={args.interval}s, "
          f"duration={'∞' if args.duration == 0 else f'{args.duration}s'}")
    print(f"[SAMPLER] Output: {args.out}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    first_write = not Path(args.out).exists()

    fh = open(args.out, "a", newline="", encoding="utf-8")
    writer = csv.writer(fh)
    if first_write:
        cols = ["timestamp", "variant", "cpu_total_pct", "cpu_proc_pct",
                "rss_mb", "temp_c"] + [f"core{i}_pct" for i in range(n_cores)]
        writer.writerow(cols)
        fh.flush()

    # Init: prima lectură de psutil.cpu_percent returnează 0; descărcăm
    psutil.cpu_percent(interval=None)
    target.cpu_percent(interval=None)

    stop = [False]

    def sigint_handler(sig, frame):
        print("\n[SAMPLER] Stop. Salvez și ies.")
        stop[0] = True

    signal.signal(signal.SIGINT, sigint_handler)

    t_start = time.time()
    try:
        while not stop[0]:
            time.sleep(args.interval)

            try:
                cpu_total = psutil.cpu_percent(interval=None)
                core_pcts = psutil.cpu_percent(interval=None, percpu=True)
                cpu_proc = target.cpu_percent(interval=None)
                rss_mb = target.memory_info().rss / (1024 * 1024)
                temp = get_cpu_temp_celsius()
            except psutil.NoSuchProcess:
                print(f"[SAMPLER] Procesul {pid} a murit. Ies.")
                break

            row = [round(time.time(), 3), args.variant,
                   round(cpu_total, 2), round(cpu_proc, 2),
                   round(rss_mb, 2),
                   round(temp, 2) if temp is not None else ""]
            row.extend(round(c, 2) for c in core_pcts)
            writer.writerow(row)
            fh.flush()

            if args.duration > 0 and (time.time() - t_start) >= args.duration:
                print("[SAMPLER] Durată atinsă. Ies.")
                break
    finally:
        fh.close()


if __name__ == "__main__":
    main()
