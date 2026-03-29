import argparse
from pathlib import Path

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd

from ids_inference import build_feature_matrix, normalize_columns, read_csv_robust


def describe_model(model_path: Path) -> None:
    model = onnx.load(model_path.as_posix())
    print("=== ONNX INPUTS ===")
    for inp in model.graph.input:
        dims = []
        for dim in inp.type.tensor_type.shape.dim:
            dims.append(dim.dim_value if dim.dim_value else dim.dim_param)
        print(inp.name, dims)


def describe_sample(df: pd.DataFrame) -> None:
    print("\n=== SAMPLE CSV ===")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))


def describe_feature_alignment(df: pd.DataFrame, feature_names_path: Path) -> None:
    wanted = np.load(feature_names_path, allow_pickle=True).tolist()
    df_num = df.select_dtypes(include=[np.number]).copy()

    missing = [col for col in wanted if col not in df_num.columns]
    extra = [col for col in df_num.columns if col not in wanted]
    same_order = list(df_num.columns) == wanted

    print("\n=== FEATURE CHECK ===")
    print("Numeric columns in sample:", len(df_num.columns))
    print("Features in feature_names.npy:", len(wanted))
    print("Missing features:", missing[:10] if missing else "none")
    print("Extra numeric columns:", extra[:10] if extra else "none")
    print("Order matches feature_names.npy:", same_order)


def describe_invalid_values(df: pd.DataFrame) -> None:
    df_num = df.select_dtypes(include=[np.number]).copy()
    arr = df_num.to_numpy(dtype=np.float64, copy=True)

    nan_mask = np.isnan(arr)
    posinf_mask = np.isposinf(arr)
    neginf_mask = np.isneginf(arr)

    print("\n=== RAW NUMERIC CHECK ===")
    print("Numeric shape before scaling:", df_num.shape)
    print("NaN count:", int(nan_mask.sum()))
    print("+Inf count:", int(posinf_mask.sum()))
    print("-Inf count:", int(neginf_mask.sum()))

    bad_columns = []
    for idx, col in enumerate(df_num.columns):
        issues = []
        if nan_mask[:, idx].any():
            issues.append(f"NaN={int(nan_mask[:, idx].sum())}")
        if posinf_mask[:, idx].any():
            issues.append(f"+Inf={int(posinf_mask[:, idx].sum())}")
        if neginf_mask[:, idx].any():
            issues.append(f"-Inf={int(neginf_mask[:, idx].sum())}")
        if issues:
            bad_columns.append(f"{col} ({', '.join(issues)})")

    if bad_columns:
        print("Columns with invalid values:")
        for item in bad_columns:
            print(" -", item)
    else:
        print("Columns with invalid values: none")


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate ONNX/scaler/features/sample artifacts together.")
    ap.add_argument("--model", default="ids_mlp_binary.onnx", help="Path to ONNX model")
    ap.add_argument("--sample", default="sample.csv", help="Path to sample CSV")
    ap.add_argument("--scaler", default="scaler.joblib", help="Path to scaler joblib")
    ap.add_argument("--features", default="feature_names.npy", help="Path to feature_names.npy")
    args = ap.parse_args()

    model_path = Path(args.model)
    sample_path = Path(args.sample)
    scaler_path = Path(args.scaler)
    feature_names_path = Path(args.features)

    describe_model(model_path)

    df_raw = read_csv_robust(sample_path)
    describe_sample(df_raw)

    df = normalize_columns(df_raw.copy())
    describe_feature_alignment(df, feature_names_path)
    describe_invalid_values(df)

    scaler = joblib.load(scaler_path)
    print("\n=== SCALER ===")
    print("n_features_in_:", getattr(scaler, "n_features_in_", "N/A"))

    X_scaled, used_features = build_feature_matrix(df, scaler, feature_names_path)
    print("\n=== SCALED FEATURES ===")
    print("Scaled shape:", X_scaled.shape)
    print("All finite after cleaning:", bool(np.isfinite(X_scaled).all()))
    print("First 5 aligned features:", used_features[:5])

    sess = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
    input_meta = sess.get_inputs()[0]
    output_meta = sess.get_outputs()[0]

    print("\n=== ONNXRUNTIME ===")
    print("Providers:", sess.get_providers())
    print("Input name:", input_meta.name)
    print("Input shape:", input_meta.shape)
    print("Output name:", output_meta.name)
    print("Output shape:", output_meta.shape)

    pred = sess.run([output_meta.name], {input_meta.name: X_scaled.astype(np.float32)})
    print("\n=== INFERENCE OK ===")
    print("Output shape:", np.array(pred[0]).shape)
    print("First rows:", pred[0][:5])


if __name__ == "__main__":
    main()
