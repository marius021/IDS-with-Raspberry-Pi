import argparse
from pathlib import Path
import numpy as np
import pandas as np
import joblib
import onnxruntime as ort

def read_csv_robust(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False, encoding="latin1", engine="python", on_bad_lines="skip")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
    )
    return df

def clean_numeric(df_num: pd.DataFrame) -> pd.DataFrame:
    # 1) Inf/-Inf -> NaN
    df_num = df_num.replace([np.inf, -np.inf], np.nan)
    # 2) Opțional: limitează outlierii aberați (protecție)
    df_num = df_num.clip(lower=-1e12, upper=1e12)
    # 3) Completează NaN cu medianele coloanelor (fallback -> 0)
    med = df_num.median(numeric_only=True)
    df_num = df_num.fillna(med).fillna(0)
    return df_num

def build_feature_matrix(df: pd.DataFrame, scaler, feature_names_path: Path | None):
    """
    Returnează (X_scaled, used_feature_names).
    - Dacă există feature_names.npy, folosește FIX acea ordine de coloane.
    - Curăță numericii (inf/-inf/NaN/outlieri) exact ca în training.
    """
    df_num = df.select_dtypes(include=[np.number]).copy()
    df_num = clean_numeric(df_num)

    if feature_names_path and feature_names_path.exists():
        wanted = np.load(feature_names_path, allow_pickle=True).tolist()
        # asigură-te că toate coloanele există; dacă lipsesc, aruncă eroare clară
        missing = [c for c in wanted if c not in df_num.columns]
        if missing:
            raise RuntimeError("Lipsesc coloane față de setul de antrenare: " + ", ".join(missing[:20]))
        df_num = df_num[wanted]
        used = wanted
    else:
        used = df_num.columns.tolist()
        if getattr(scaler, "n_features_in_", None) and len(used) != scaler.n_features_in_:
            raise RuntimeError(
                f"Numărul de coloane numerice ({len(used)}) nu corespunde cu scaler.n_features_in_ "
                f"({scaler.n_features_in_}). Folosește --features feature_names.npy."
            )

    # acum toate valorile sunt finite și rezonabile
    X = df_num.values.astype(np.float32)
    X_scaled = scaler.transform(X)
    return X_scaled, used

def run_inference(input_csv, model_path, scaler_path, out_csv, mode, feature_names_path, classes_path, threshold=0.5):
    df = read_csv_robust(input_csv)
    df = normalize_columns(df)

    scaler = joblib.load(scaler_path)
    X_scaled, used_features = build_feature_matrix(df, scaler, feature_names_path)

    sess = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: X_scaled})

    if mode == "binary":
        proba = outputs[0].ravel().astype(float)
        pred = (proba >= threshold).astype(int)
        df["pred_attack_proba"] = proba
        df["pred_label"] = np.where(pred == 1, "ATTACK", "BENIGN")
    else:
        logits = outputs[0]
        pred_idx = np.argmax(logits, axis=1)
        if classes_path and classes_path.exists():
            classes = np.load(classes_path, allow_pickle=True).tolist()
            classes = list(classes)[: logits.shape[1]]
            pred_names = [classes[i] for i in pred_idx]
        else:
            pred_names = [f"class_{i}" for i in pred_idx]
        df["pred_class_idx"] = pred_idx
        df["pred_class"] = pred_names

    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[OK] Predicțiile au fost scrise în: {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV de intrare pentru inferență")
    ap.add_argument("--model", required=True, help="Model ONNX (ids_mlp_binary.onnx sau ids_mlp_multiclass.onnx)")
    ap.add_argument("--scaler", default="scaler.joblib", help="Scaler joblib salvat la training")
    ap.add_argument("--mode", choices=["binary", "multi"], required=True, help="Tipul modelului")
    ap.add_argument("--out", default="predictions.csv", help="CSV de ieșire cu predicții")
    ap.add_argument("--features", default="feature_names.npy", help="Lista de coloane din training")
    ap.add_argument("--classes", default="classes.npy", help="(doar multi) fișier cu numele claselor")
    ap.add_argument("--threshold", type=float, default=0.5, help="(binary) pragul ATTACK")
    args = ap.parse_args()

    run_inference(
        input_csv=Path(args.input),
        model_path=Path(args.model),
        scaler_path=Path(args.scaler),
        out_csv=Path(args.out),
        mode=args.mode,
        feature_names_path=Path(args.features) if args.features else None,
        classes_path=Path(args.classes) if args.classes else None,
        threshold=args.threshold,
    )

if __name__ == "__main__":
    main()