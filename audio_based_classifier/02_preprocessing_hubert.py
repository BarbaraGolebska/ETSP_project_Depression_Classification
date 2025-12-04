import numpy as np
import os
import pandas as pd

# Path to the processed embeddings directory
FOLDER = r"C:\Users\ninas\OneDrive\Documentos\Master\AI\1st Semester\Essentials Text and Speech\ESTP_project-2\data\embeddings\processed"

# Path to the labels CSV
LABELS_PATH = r"data/raw/labels/detailed_labels.csv"

embeddings = []
participant_ids = []

# List all .npy files
files = [f for f in os.listdir(FOLDER) if f.endswith('.npy')]
files.sort()  # recommended for reproducibility

print("Found files:", len(files))

for fname in files:
    path = os.path.join(FOLDER, fname)

    # ---- Extract participant ID ----
    participant_id = fname.split("_embedding")[0]

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
labels_df = pd.read_csv(LABELS_PATH).rename(columns={
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
output_path = "data/processed/hubert_aggregated_embeddings.csv"
df.to_csv(output_path, index=False)

print("Final DataFrame shape:", df.shape)
print("Saved CSV to:", output_path)
