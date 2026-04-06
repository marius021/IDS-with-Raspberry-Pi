import numpy as np
from pathlib import Path

feats = np.load("feature_names.npy", allow_pickle=True).tolist()
feats = [str(x).strip() for x in feats]
Path("feature_names.txt").write_text("\n".join(feats), encoding="utf-8")
print("Am generat feature_names.txt")