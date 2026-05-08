#!/usr/bin/env python3
"""
feed_csv.py

Simulates a growing live CSV by streaming rows from a source file.
Use this to feed ips_hailo.py or ips_realtime_v2.py during real-time testing.

Usage:
  # Basic: stream sample.csv into live_sample.csv at 50 rows every 4 seconds
  python3 feed_csv.py --src sample.csv --dst live_sample.csv

  # Faster feed for quick tests
  python3 feed_csv.py --src sample.csv --dst live_sample.csv --chunk 10 --delay 1

  # Loop indefinitely (restarts from the beginning when source is exhausted)
  python3 feed_csv.py --src sample.csv --dst live_sample.csv --loop
"""

import argparse
import time
from pathlib import Path

import pandas as pd

DEFAULT_BASE = Path.home() / "Desktop" / "IDS-with-Raspberry-Pi" / "ids_test"


def main():
    ap = argparse.ArgumentParser(
        description="Stream source CSV rows into a growing live CSV for real-time IPS testing."
    )
    ap.add_argument("--src", default=str(DEFAULT_BASE / "sample_big.csv"),
                    help="Source CSV to stream from")
    ap.add_argument("--dst", default=str(DEFAULT_BASE / "live_sample.csv"),
                    help="Destination (live) CSV that ips_hailo.py monitors")
    ap.add_argument("--chunk", type=int, default=50,
                    help="Number of rows to append per tick (default: 50)")
    ap.add_argument("--delay", type=float, default=4.0,
                    help="Seconds between ticks (default: 4)")
    ap.add_argument("--loop", action="store_true",
                    help="Restart from the beginning when source is exhausted")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        raise FileNotFoundError(f"Source CSV not found: {src}")

    df = pd.read_csv(src)
    print(f"[FEED] Loaded {len(df)} rows from {src}")
    print(f"[FEED] Destination : {dst}")
    print(f"[FEED] Chunk size  : {args.chunk} rows")
    print(f"[FEED] Delay       : {args.delay}s")
    print(f"[FEED] Loop        : {args.loop}")

    while True:
        df.iloc[:0].to_csv(dst, index=False)
        print(f"[FEED] Created {dst.name} with header only")

        start = 0
        while start < len(df):
            end = min(start + args.chunk, len(df))
            df.iloc[start:end].to_csv(dst, mode="a", index=False, header=False)
            print(f"[FEED] Appended rows {start}:{end}  (total in file: {end})")
            start = end
            time.sleep(args.delay)

        print("[FEED] Source exhausted.")
        if not args.loop:
            break
        print("[FEED] Looping — restarting from the beginning...")

    print("[FEED] Done.")


if __name__ == "__main__":
    main()
