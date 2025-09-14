#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ========= Helpers robuste pentru CSV =========

def read_csv_robust(path: Path) -> pd.DataFrame:
    """Citește CSV cu încercări pe encodări uzuale; sare peste linii corupte ca ultim resort."""
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except UnicodeDecodeError:
            continue
    try:
        return pd.read_csv(path, low_memory=False, encoding="latin1",
                           engine="python", on_bad_lines="skip")
    except TypeError:
        return pd.read_csv(path, low_memory=False, encoding="latin1",
                           engine="python", error_bad_lines=False, warn_bad_lines=True)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizează header-ele: elimină BOM, spații, lowercase."""
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
    )
    return df


def find_label_col(cols) -> str:
    """Identifică numele efectiv al coloanei de etichetă (label/attack/etc.)."""
    base = [c.strip().lower() for c in cols]
    for cand in ("label", "class", "attack", "category", "result"):
        if cand in base:
            return cand
    for c in base:
        if "label" in c or "attack" in c:
            return c
    raise KeyError(f"Nu găsesc coloana etichetă. Primele coloane: {list(cols)[:10]}")


def _discover_csvs(data_dir: Path) -> list[Path]:
    """Caută CSV-urile uzuale din CICIDS2017."""
    patterns = [
        "TrafficLabelling_*.csv",
        "TrafficLabelling-*.csv",
        "*pcap_ISCX.csv",
        "*.csv",
    ]
    found = []
    for pat in patterns:
        found.extend(sorted(data_dir.glob(pat)))
    uniq, seen = [], set()
    for p in found:
        if p.name not in seen:
            uniq.append(p)
            seen.add(p.name)
    return uniq


def load_data(data_dir: Path, csv_files: list[str] | None = None) -> pd.DataFrame:
    """Încarcă toate CSV-urile și unifică eticheta la 'label'."""
    data_dir = Path(data_dir)
    paths = [data_dir / f for f in csv_files] if csv_files else _discover_csvs(data_dir)
    if not paths:
        raise FileNotFoundError(f"Niciun CSV găsit în {data_dir}")

    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Lipsesc fișiere în {data_dir}: {missing}")

    dfs = []
    for p in paths:
        print(f"[INFO] Încarc {p.name} ...")
        df = read_csv_robust(p)
        df = normalize_columns(df)
        lab_col = find_label_col(df.columns)
        if lab_col != "label":
            df = df.rename(columns={lab_col: "label"})
        df["label"] = df["label"].astype(str).str.strip()
        df["__source_file"] = p.name
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True, sort=False)


def detect_time_col(df: pd.DataFrame) -> str | None:
    cols = [c.lower() for c in df.columns]
    for cand in ["timestamp", "flow start", "starttime", "start time", "flowstart", "start"]:
        if cand in cols:
            return df.columns[cols.index(cand)]
    return None


def clean_and_sort(df: pd.DataFrame):
    """Curățare numerică + sortare temporală."""
    df = df.copy()
    time_col = detect_time_col(df)
    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    else:
        df["__synthetic_time"] = pd.date_range("2017-06-01", periods=len(df), freq="S")
        time_col = "__synthetic_time"

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
        for c in num_cols:
            med = df[c].median()
            df[c] = df[c].fillna(med if not np.isnan(med) else 0)

    df = df.sort_values(time_col).reset_index(drop=True)
    return df, time_col


def time_split(X_df: pd.DataFrame, y_bin: np.ndarray, train=0.70, val=0.15):
    n = len(X_df)
    tr_end = int(train * n)
    vl_end = int((train + val) * n)
    X_train = X_df.iloc[:tr_end].values
    X_val = X_df.iloc[tr_end:vl_end].values
    X_test = X_df.iloc[vl_end:].values
    y_train = y_bin[:tr_end]
    y_val = y_bin[tr_end:vl_end]
    y_test = y_bin[vl_end:]
    return (X_train, X_val, X_test, y_train, y_val, y_test)


def build_mlp(input_dim: int, out_dim: int = 1) -> nn.Module:
    if out_dim == 1:
        return nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid(),
        )
    else:
        return nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, out_dim),
        )


def train_binary(X_train, X_val, X_test, y_train, y_val, y_test, epochs=12, batch_size=2048, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                        torch.tensor(y_train, dtype=torch.float32)),
                          batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                      torch.tensor(y_val, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                       torch.tensor(y_test, dtype=torch.float32)),
                         batch_size=batch_size, shuffle=False)

    model = build_mlp(X_train.shape[1], 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    best_state, best_val = None, 1e9
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss, nobs = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * xb.size(0); nobs += xb.size(0)
        tr = tr_loss / max(nobs, 1)

        model.eval(); val_loss, nobs = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_loss += loss.item() * xb.size(0); nobs += xb.size(0)
        vl = val_loss / max(nobs, 1)
        print(f"[Binary] Epoch {ep:02d} | train_loss={tr:.4f} | val_loss={vl:.4f}")
        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval(); preds, ys = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            proba = model(xb.to(device)).cpu().numpy().ravel()
            preds.extend((proba > 0.5).astype(int)); ys.extend(yb.numpy())
    print("\n[BINARY] Classification report (TEST):")
    print(classification_report(ys, preds, digits=4))

    dummy = torch.randn(1, X_train.shape[1], dtype=torch.float32).to(device)
    torch.onnx.export(model, dummy, "ids_mlp_binary.onnx",
                      input_names=["input"], output_names=["prob"],
                      opset_version=12,
                      dynamic_axes={"input": {0: "batch"}, "prob": {0: "batch"}})
    print("Salvat: ids_mlp_binary.onnx")


def train_multiclass(X_train, X_val, X_test, labels_series, epochs=12, batch_size=2048, lr=1e-3):
    # pregătește etichetele multiclass
    labels = labels_series.astype(str).str.strip()
    labels = labels.str.replace(r"^Web Attack.*", "WebAttack", regex=True)

    le = LabelEncoder()
    y_all = le.fit_transform(labels.values)
    classes = list(le.classes_)
    np.save("classes.npy", np.array(classes, dtype=object))
    joblib.dump(le, "label_encoder.joblib")
    print(f"Clase: {len(classes)} (ex: {classes[:10]})")

    tr_end = len(X_train)
    vl_end = tr_end + len(X_val)
    y_train, y_val, y_test = y_all[:tr_end], y_all[tr_end:vl_end], y_all[vl_end:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                        torch.tensor(y_train, dtype=torch.long)),
                          batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                      torch.tensor(y_val, dtype=torch.long)),
                        batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                       torch.tensor(y_test, dtype=torch.long)),
                         batch_size=batch_size, shuffle=False)

    model = build_mlp(X_train.shape[1], len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_state, best_val = None, 1e9
    for ep in range(1, epochs + 1):
        model.train(); tr_loss, nobs = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = loss_fn(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * xb.size(0); nobs += xb.size(0)
        tr = tr_loss / max(nobs, 1)

        model.eval(); val_loss, nobs = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = loss_fn(out, yb)
                val_loss += loss.item() * xb.size(0); nobs += xb.size(0)
        vl = val_loss / max(nobs, 1)
        print(f"[Multi]  Epoch {ep:02d} | train_loss={tr:.4f} | val_loss={vl:.4f}")
        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    # ===== VARIANTA A: raportează doar clasele prezente în TEST (și/sau în predicții) =====
    model.eval(); preds, ys = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            logits = model(xb.to(device)).cpu().numpy()
            preds.extend(np.argmax(logits, axis=1)); ys.extend(yb.numpy())
    preds = np.asarray(preds)
    ys = np.asarray(ys)

    present = np.unique(np.concatenate([ys, preds]))     # toate clasele observate
    names_present = [classes[i] for i in present]

    print("\n[MULTI-CLASS] Classification report (TEST):")
    print(classification_report(
        ys, preds,
        labels=present,                 # <<— IMPORTANT
        target_names=names_present,     # sincron cu 'labels'
        digits=4,
        zero_division=0
    ))

    dummy = torch.randn(1, X_train.shape[1], dtype=torch.float32).to(device)
    torch.onnx.export(model, dummy, "ids_mlp_multiclass.onnx",
                      input_names=["input"], output_names=["logits"],
                      opset_version=12,
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    print("Salvat: ids_mlp_multiclass.onnx")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Folderul cu CSV-urile CIC-IDS2017 (flow-level)")
    parser.add_argument("--epochs", type=int, default=12, help="Numărul de epoci.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("Încarc datele din:", data_dir)
    raw = load_data(data_dir)
    print("Shape brut:", raw.shape)

    df, time_col = clean_and_sort(raw)
    print("După curățare/sortare:", df.shape, "| coloană timp:", time_col)

    # Feature matrix (numerice) — exclude etichete/metadate
    drop_cols = ["label", "__source_file"]
    X_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X_df = X_df.select_dtypes(include=[np.number]).copy()
    if X_df.shape[1] == 0:
        raise RuntimeError("Nu au rămas coloane numerice după filtrare.")
    np.save("feature_names.npy", np.array(X_df.columns.tolist(), dtype=object))
    print("[INFO] Salvat feature_names.npy cu", len(X_df.columns), "coloane.")


    # Binary labels
    y_bin = (df["label"].str.upper() != "BENIGN").astype(int).values

    # Split pe timp
    X_train, X_val, X_test, y_train, y_val, y_test = time_split(X_df, y_bin)

    # Scale
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, "scaler.joblib")
    print("Scaler salvat: scaler.joblib")

    # Train & export
    train_binary(X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, epochs=args.epochs)
    train_multiclass(X_train_s, X_val_s, X_test_s, df["label"], epochs=args.epochs)

    print("\nFișiere generate:")
    print(" - scaler.joblib")
    print(" - ids_mlp_binary.onnx")
    print(" - ids_mlp_multiclass.onnx")
    print(" - label_encoder.joblib / classes.npy")


if __name__ == "__main__":
    main()
