import pandas as pd
import time
from pathlib import Path

BASE = Path("/home/maurice/Desktop/IDS-with-Raspberry-Pi/ids_test")
SRC = BASE / "sample_big.csv"
DST = BASE / "live_sample.csv"

CHUNK_SIZE = 50
DELAY_SEC = 4

df = pd.read_csv(SRC)

# scrie header doar o singură dată
df.iloc[:0].to_csv(DST, index=False)
print(f"[FEED] creat {DST} cu header only")

start = 0
while start < len(df):
    end = min(start + CHUNK_SIZE, len(df))
    df.iloc[start:end].to_csv(DST, mode="a", index=False, header=False)
    print(f"[FEED] adaugat randurile {start}:{end}")
    start = end
    time.sleep(DELAY_SEC)

print("[FEED] terminat")