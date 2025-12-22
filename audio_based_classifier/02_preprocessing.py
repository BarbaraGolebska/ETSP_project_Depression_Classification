import numpy as np
import pandas as pd 
import os
from sklearn.model_selection import train_test_split
from pathlib import Path
os.chdir(Path(__file__).resolve().parents[1])


def aggregate_features(df):
    # assume first column = timestamp
    X = df.iloc[:, 2:]
    summary = pd.Series({
        **{f"{col}_mean": X[col].mean() for col in X.columns},
        **{f"{col}_std": X[col].std() for col in X.columns},
    })
    return summary

def preprocess_hubert():
    
    # Path to the processed embeddings directory
    folder = r"data/raw/features"

    # Path to the labels CSV
    labels_path = r"data/raw/labels/detailed_labels.csv"

    embeddings = []
    participant_ids = []

    # List all .npy files
    files = [f for f in os.listdir(folder) if f.endswith('.npy')]
    files.sort()  # recommended for reproducibility

    print("Found files:", len(files))

    for fname in files:
        path = os.path.join(folder, fname)

        # ---- Extract participant ID ----
        participant_id = fname.split("_hubert_embedding")[0]

        # ---- Load the embedding ----
        arr = np.load(path)  # shape: (768,)
        print("Processing:", participant_id, "Shape:", arr.shape)

        embeddings.append(arr)
        participant_ids.append(participant_id)

    # ---- Convert to DataFrame ----
    emb_matrix = np.vstack(embeddings)

    # Column names: feat0, feat1, ..., featN
    n_features = emb_matrix.shape[1]
    columns = [f"feat{i}" for i in range(n_features)]

    df = pd.DataFrame(emb_matrix, columns=columns)
    df.insert(0, "participant_id", participant_ids)

    # ---- Load labels ----
    labels_df = pd.read_csv(labels_path).rename(columns={
        "Participant": "participant_id",
        "Depression_label": "target_depr",
        "PTSD_label": "target_ptsd"
    })
    labels_df["participant_id"] = labels_df["participant_id"].astype(str)

    # ---- Merge embeddings with labels ----
    df = df.merge(labels_df[["participant_id", "target_depr", "target_ptsd", "split"]], 
                on="participant_id", how="inner")

    # ---- Save CSV ----
    #df = df.set_index("participant_id")
    output_path = "data/processed/audio/hubert/hubert_aggregated_embeddings.csv"
    df.to_csv(output_path, index=False)

    print("Final DataFrame shape:", df.shape)
    print("Saved CSV to:", output_path)


def main(chosen_ftype):
    labels_df = pd.read_csv('data/raw/labels/detailed_labels.csv').rename(columns={"Participant": "participant_id", "Depression_label": "target_depr", "PTSD_label": "target_ptsd"})
    labels_df["participant_id"] = labels_df["participant_id"].astype(str)

    participant_list = labels_df["participant_id"].tolist()


    agg_features = []
    for pid in participant_list:
        print(f"Processing patient: {pid}")
        all_ftypes = {
            "bow_egemaps": f"{pid}_BoAW_openSMILE_2.3.0_eGeMAPS.csv",
            "bow_mfcc": f"{pid}_BoAW_openSMILE_2.3.0_MFCC.csv",
            "densenet201": f"{pid}_densenet201.csv",
            "vgg16": f"{pid}_vgg16.csv",
            "ek_egemaps":f"{pid}_OpenSMILE2.3.0_egemaps.csv",
            "ek_mfcc":f"{pid}_OpenSMILE2.3.0_mfcc.csv"
        }
        ftypes = {ftype:path for ftype, path in all_ftypes.items() if ftype == chosen_ftype}

        patient_summary = []
        for ftype, path in ftypes.items():
            if ftype in ["ek_egemaps", "ek_mfcc"]:
                df = pd.read_csv(f"data/raw/features/{path}",  sep=';')

            elif ftype in ["densenet201", "vgg16"]:
                df = pd.read_csv(f"data/raw/features/{path}",  sep=',', decimal='.')
                

            else:
                df = pd.read_csv(f"data/raw/features/{path}", sep='[,;]', engine='python', header=None)
            agg = aggregate_features(df)
            agg.index = [f"{ftype}_{col}" for col in agg.index]
            patient_summary.append(agg)

        all_feats = pd.concat(patient_summary)
        all_feats["participant_id"] = pid
        agg_features.append(all_feats)

    df_features = pd.DataFrame(agg_features).set_index("participant_id")

    data = df_features.merge(labels_df[["participant_id", "target_depr", "target_ptsd", "split"]],
                                    on="participant_id", how="inner")
    
    data = data.set_index("participant_id")
    data.to_csv(f"data/processed/{chosen_ftype}_aggregated_features.csv")

if __name__ == "__main__":
    ftypes_list = ["bow_egemaps", "bow_mfcc", "densenet201", "vgg16", "ek_egemaps", "ek_mfcc"]
    for ftype in ftypes_list:
        main(ftype)
    preprocess_hubert()