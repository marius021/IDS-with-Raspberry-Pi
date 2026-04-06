import joblib
import numpy as np
from pathlib import Path

scaler = joblib.load("scaler.joblib")

np.savez(
    "scaler_params.npz",
    mean=np.asarray(scaler.mean_, dtype=np.float32),
    scale=np.asarray(scaler.scale_, dtype=np.float32),
    var=np.asarray(scaler.var_, dtype=np.float32),
    n_features_in=np.asarray([scaler.n_features_in_], dtype=np.int32),
    with_mean=np.asarray([int(getattr(scaler, "with_mean", True))], dtype=np.int32),
    with_std=np.asarray([int(getattr(scaler, "with_std", True))], dtype=np.int32),
)

print("Am generat scaler_params.npz")