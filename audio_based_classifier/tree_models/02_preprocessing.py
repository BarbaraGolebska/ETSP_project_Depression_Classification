import numpy as np
import pandas as pd
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)

splits = pd.read_csv("../data/raw/labels/detailed_labels.csv")

splits["Participant"] = splits["Participant"].astype(str)
splits = splits.rename(columns={"Participant": "Participant_ID"})
splits = splits.rename(columns={"Depression_label": "PHQ_Binary"})

label_col = "PHQ_Binary"

X_train, y_train = [], []
X_dev, y_dev = [], []
X_test, y_test = [], []

for _, row in splits.iterrows():

    pid = row["Participant_ID"]
    label = row[label_col]
    split = row["split"]

    emb_path = Path(f"../data/processed/{pid}_embedding.npy")
    eg_path = Path(f"../data/raw/features/{pid}_OpenSMILE2.3.0_egemaps.csv")
    mfcc_path = Path(f"../data/raw/features/{pid}_OpenSMILE2.3.0_mfcc.csv")

    if not emb_path.exists() or not eg_path.exists() or not mfcc_path.exists():
        logging.warning(f"Missing files for PID {pid}")
        continue

    emb_raw = np.load(emb_path)
    emb = emb_raw.mean(axis=0) if emb_raw.ndim == 2 else emb_raw

    # Averaged over time
    eg = pd.read_csv(eg_path, sep=";")
    eg = eg.drop(columns=["name", "frameTime"])
    eg = eg.apply(pd.to_numeric, errors="coerce").mean().values
    mfcc = pd.read_csv(mfcc_path, sep=";").apply(pd.to_numeric, errors="coerce").mean().values

    features = np.concatenate([emb, eg,mfcc])

    if split == "train":
        X_train.append(features)
        y_train.append(label)

    elif split == "dev":
        X_dev.append(features)
        y_dev.append(label)

    elif split == "test":
        X_test.append(features)
        y_test.append(label)

    else:
        logging.error(f"Unknown split value '{split}' for PID={pid}")

X_train = np.vstack(X_train)
X_dev = np.vstack(X_dev)
X_test = np.vstack(X_test)

y_train = np.array(y_train)
y_dev = np.array(y_dev)
y_test = np.array(y_test)

out_dir = Path("../data/processed/audio/lightgbm_smote_hubert_mfcc_egemaps")
out_dir.mkdir(parents=True, exist_ok=True)

np.save(out_dir / "X_train.npy", X_train)
np.save(out_dir / "y_train.npy", y_train)

np.save(out_dir / "X_dev.npy", X_dev)
np.save(out_dir / "y_dev.npy", y_dev)

np.save(out_dir / "X_test.npy", X_test)
np.save(out_dir / "y_test.npy", y_test)

logging.info("\nPreprocessing complete.")
logging.info("Shapes:")
logging.info(f"Train: {X_train.shape}, {y_train.shape}")
logging.info(f"Dev:   {X_dev.shape}, {y_dev.shape}")
logging.info(f"Test:  {X_test.shape}, {y_test.shape}")