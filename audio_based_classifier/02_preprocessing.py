import numpy as np
import pandas as pd
from pathlib import Path


meta = pd.read_csv("../data/raw/metadata_mapped.csv")
X_list = []
y_list = []

for _, row in meta.iterrows():
    pid = row["Participant_ID"]
    print(repr(pid))
    emb_path = Path(f"../data/processed/{pid}_embedding.npy")
    eg_path = Path(f"../data/raw/features/{pid}_OpenSMILE2.3.0_egemaps.csv")
    mfcc_path = Path(f"../data/raw/features/{pid}_OpenSMILE2.3.0_mfcc.csv")
    if not emb_path.exists() or not eg_path.exists() or not mfcc_path.exists():
        continue
    
    emb_raw = np.load(emb_path)

    if emb_raw.ndim == 2:
        emb = emb_raw.mean(axis=0)
    else:
        emb = emb_raw

    eg = pd.read_csv(eg_path, sep=";").apply(pd.to_numeric, errors="coerce").mean().values
    mfcc = pd.read_csv(mfcc_path, sep=";").apply(pd.to_numeric, errors="coerce").mean().values
    
    print(f"PID: {pid}")

    print("emb type:", type(emb), "shape:", getattr(emb, 'shape', None))
    print("eg type:", type(eg), "shape:", getattr(eg, 'shape', None))
    print("mfcc type:", type(mfcc), "shape:", getattr(mfcc, 'shape', None))
   
    features = np.concatenate([emb, eg, mfcc])
    X_list.append(features)
    y_list.append(row["PHQ_Binary"])

X = np.vstack(X_list)
y = np.array(y_list)

np.save("../data/vectors/X_features.npy", X)
np.save("../data/vectors/y_labels.npy", y)
