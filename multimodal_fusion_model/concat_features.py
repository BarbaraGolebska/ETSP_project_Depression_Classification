import numpy as np
import os
import pandas as pd

def load_parquet_embeddings():
    # ==========================================
    # CONFIGURATION
    # ==========================================
    # Path to the input Parquet file (Text Embeddings)
    INPUT_PARQUET = r"data/processed/embeddings.parquet"

    # Path to the labels CSV
    LABELS_PATH = r"data/raw/labels/detailed_labels.csv"

    # Output path
    OUTPUT_PATH = r"data/processed/text_embeddings_aggregated.csv"

    # ==========================================
    # PROCESSING
    # ==========================================

    print("Loading Parquet file...")
    df_parquet = pd.read_parquet(INPUT_PARQUET)

    # Ensure the embedding column is treated as numpy arrays
    # (Based on the snippet you provided)
    df_parquet["embedding"] = df_parquet["embedding"].apply(lambda x: np.array(x))

    print(f"Loaded Parquet with shape: {df_parquet.shape}")
    print(f"Columns found: {df_parquet.columns.tolist()}")

    # Check if participant_id exists in the parquet
    if "participant_id" not in df_parquet.columns:
        # If your parquet uses 'Participant' or 'id', rename it here
        if "Participant" in df_parquet.columns:
            df_parquet = df_parquet.rename(columns={"Participant": "participant_id"})
        elif "id" in df_parquet.columns:
            df_parquet = df_parquet.rename(columns={"id": "participant_id"})
        else:
            raise ValueError("Error: Could not find 'participant_id' column in the Parquet file.")

    # ---- 1. Expand Embeddings into Columns ----
    # Stack the arrays into a matrix (N_samples, 768)
    emb_matrix = np.vstack(df_parquet["embedding"].values)

    print(f"Embedding Matrix Shape: {emb_matrix.shape}") # Should be (N, 768)

    # Create column names: text_feat0, text_feat1... 
    # DISTINCT names are crucial for fusion later!
    n_features = emb_matrix.shape[1]
    columns = [f"text_feat{i}" for i in range(n_features)]

    # Create the Features DataFrame
    df_features = pd.DataFrame(emb_matrix, columns=columns)

    # Add the participant_id back (to ensure we can merge)
    df_features.insert(0, "participant_id", df_parquet["participant_id"].astype(str))

    # ---- 2. Load Labels ----
    print("Loading Labels...")
    labels_df = pd.read_csv(LABELS_PATH).rename(columns={
        "Participant": "participant_id",
        "Depression_label": "target_depr",
        "PTSD_label": "target_ptsd"
    })
    labels_df["participant_id"] = labels_df["participant_id"].astype(str)

    # ---- 3. Merge Embeddings with Labels ----
    # Inner join to ensure we only keep participants that have both Embeddings AND Labels
    df_final = df_features.merge(labels_df[["participant_id", "target_depr", "target_ptsd", "split"]], 
                                on="participant_id", how="inner")

    # ---- 4. Save CSV ----
    df_final.to_csv(OUTPUT_PATH, index=False)

    print("-" * 30)
    print("Final DataFrame shape:", df_final.shape)
    print("Saved Text Embeddings to:", OUTPUT_PATH)

def concat_features_large():

    # Load your datasets
    df1 = pd.read_csv("data/processed/text_embeddings_aggregated.csv")
    df2 = pd.read_csv("data/processed/hubert_aggregated_embeddings.csv")
    df3 = pd.read_csv("data/processed/ExpertK_aggregated_features.csv") # this file is the aggregation of OpenSmile mfcc + egemaps


    p1 = set(df1['participant_id'])
    p2 = set(df2['participant_id'])
    p3 = set(df3['participant_id'])

    # Check if all are equal
    all_same = (p1 == p2 == p3)
    print("All datasets have the same participants?", all_same)

    # Optional: see differences
    print("In df1 but not df2:", p1 - p2)
    print("In df2 but not df3:", p2 - p3)
    print("In df3 but not df1:", p3 - p1)




    # Columns to merge on
    merge_on = ["participant_id", "target_depr", "target_ptsd", "split"]

    # Merge df1 and df2 first
    df_12 = pd.merge(df1, df2, on=merge_on, how='inner')

    # Merge the result with df3
    df_all = pd.merge(df_12, df3, on=merge_on, how='inner')

    # Optional: check the result
    print(df_all.head())
    print(df_all.shape)
    # Save the merged dataframe
    df_all.to_csv("data/processed/expertk_hubert_text_concat_early_fusion.csv", index=False)

def concat_features():
    # Load your datasets
    df1 = pd.read_csv("data/processed/text_embeddings_aggregated.csv")
    df2 = pd.read_csv("data/processed/hubert_aggregated_embeddings.csv")
    

    p1 = set(df1['participant_id'])
    p2 = set(df2['participant_id'])


    # Check if all are equal
    all_same = (p1 == p2)
    print("All datasets have the same participants?", all_same)

    # Optional: see differences
    print("In df1 but not df2:", p1 - p2)

    # Columns to merge on
    merge_on = ["participant_id", "target_depr", "target_ptsd", "split"]

    # Merge df1 and df2 first
    df_all = pd.merge(df1, df2, on=merge_on, how='inner')


    # Optional: check the result
    print(df_all.head())
    print(df_all.shape)
    # Save the merged dataframe
    df_all.to_csv("data/processed/concat_early_fusion.csv", index=False)

# --- Example of running the model ---
if __name__ == "__main__":

    concat_features_large()  # Ensure the concatenated CSV is created