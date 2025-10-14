import time
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib   
import onnxruntime as ort

# ---------- Config implicită (poți suprascrie din CLI) ----------
DEFAULT_INPUT = Path.home() / "ids" / "sample.csv"       # fișierul care “crește” sau CSV periodic
ART = Path.home() / "ids" / "artifacts"                  # unde ai pus artefactele pe Pi
DEFAULT_MODEL = ART / "ids_mlp_binary.onnx"
DEFAULT_SCALER = ART / "scaler.joblib"
DEFAULT_FEATS  = ART / "feature_names.npy"
ALERT_LOG = Path.home() / "ids" / "alerts.log"
POLL_SEC = 3                                             # cat de des verificam

THRESHOLD = 0.5                                          # prag pentru ATTACK (binary)
BATCH_SIZE = 1024


# ---------- Utilitare ----------
def clean_numeric(df_num: pd.DataFrame) -> pd.DataFrame:
    df_num = df_num.replace([np.inf, -np.inf], np.nan)
    # limitează outlierii exagerați
    df_num = df_num.clip(lower=-1e12, upper=1e12)
    # completează lipsurile cu medianele pe coloană (fallback -> 0)
    med = df_num.median(numeric_only=True)
    df_num = df_num.fillna(med).fillna(0)
    return df_num

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
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
    missing = [c for c in wanted if c not in df_num.columns]
    if missing:
        raise RuntimeError("Lipsesc coloane față de setul de antrenare: " + ", ".join(missing[:20]))
    df_num = df_num[wanted]

    X = df_num.values.astype(np.float32)
    Xs = scaler.transform(X)
    return Xs

def run_batch(sess, input_name, Xs):
    outs = sess.run(None, {input_name: Xs})
    proba = outs[0].ravel().astype(float)  # model binary (sigmoid)
    pred  = (proba >= THRESHOLD).astype(int)
    return proba, pred

def append_alerts(rows_df: pd.DataFrame, prob, pred):
    ts = int(time.time())
    with open(ALERT_LOG, "a", encoding="utf-8") as f:
        for i, is_attack in enumerate(pred):
            if is_attack:
                rec = {
                    "ts": ts,
                    "prob": float(prob[i]),
                    "row_index": int(rows_df.index[i]),
                }
                f.write(json.dumps(rec) + "\n")
    # și pe consolă, un sumar
    attacks = int(pred.sum())
    if attacks:
        print(f"[ALERT] {attacks} eveniment(e) ATTACK (top prob: {prob[pred==1].max():.3f})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT), help="CSV sursă (fișier unic care crește)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="Model ONNX (binary)")
    ap.add_argument("--scaler", default=str(DEFAULT_SCALER), help="scaler.joblib")
    ap.add_argument("--features", default=str(DEFAULT_FEATS), help="feature_names.npy")
    ap.add_argument("--poll", type=int, default=POLL_SEC, help="interval de poll (sec)")
    ap.add_argument("--batch", type=int, default=BATCH_SIZE, help="mărimea batch-ului")
    args = ap.parse_args()

    input_csv = Path(args.input)
    feats_p   = Path(args.features)

    # Încarcă artefacte
    scaler = joblib.load(args.scaler)
    sess = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # Ține minte ultimul rând procesat (pentru fișierul care crește)
    last_seen = 0
    print(f"[INFO] Pornit. Monitorizez: {input_csv} | poll={args.poll}s | batch={args.batch}")

    while True:
        if input_csv.exists():
            try:
                df_all = pd.read_csv(input_csv, low_memory=False)
                n = len(df_all)
                if n > last_seen:
                    # prelucrează doar rândurile noi
                    df_new = df_all.iloc[last_seen:n]
                    # procesare în batch-uri
                    for start in range(0, len(df_new), args.batch):
                        chunk = df_new.iloc[start:start+args.batch]
                        Xs = build_feature_matrix(chunk, scaler, feats_p)
                        prob, pred = run_batch(sess, input_name, Xs)
                        append_alerts(chunk, prob, pred)
                    last_seen = n
            except Exception as e:
                print(f"[WARN] Eroare la procesare: {e}")
        else:
            print("[INFO] Aștept fișierul de intrare...")

        time.sleep(args.poll)

if __name__ == "__main__":
    main()
