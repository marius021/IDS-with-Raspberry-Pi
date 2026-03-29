import pandas as pd
from sklearn.metrics import classification_report
df = pd.read_csv("preds.csv")
print(classification_report((df["label"].str.upper()!="BENIGN").astype(int),
                            (df["pred_label"]!="BENIGN").astype(int), digits=4))