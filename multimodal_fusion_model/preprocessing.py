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



import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MultimodalDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        
        # --- Separate Feature Columns ---
        # Identify text columns (created in your script as text_feat0...)
        self.text_cols = [c for c in self.df.columns if "text_feat" in c]
        
        # Identify audio columns (Assuming you named them audio_feat... or hubert_feat...)
        # If you didn't name them specifically, you might filter by exclusion or specific indices.
        self.audio_cols = [c for c in self.df.columns if "audio_feat" in c] # Update string match if needed!
        
        # Targets
        self.targets = self.df["target_depr"].values
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Extract Text Features
        text_data = self.df.iloc[idx][self.text_cols].values.astype(np.float32)
        
        # Extract Audio Features
        audio_data = self.df.iloc[idx][self.audio_cols].values.astype(np.float32)
        
        # Extract Label
        label = self.targets[idx].astype(np.float32)
        
        return {
            "text": torch.tensor(text_data),
            "audio": torch.tensor(audio_data),
            "label": torch.tensor(label)
        }
    
import torch
import torch.nn as nn

class LowRankBilinearFusion(nn.Module):
    def __init__(self, text_dim, audio_dim, rank=128, output_dim=1):
        """
        Args:
            text_dim (int): Number of text features (e.g., 768).
            audio_dim (int): Number of audio features (e.g., 768 or 1024).
            rank (int): The shared projection dimension (hyperparameter).
            output_dim (int): 1 for binary classification, >1 for multi-class.
        """
        super(LowRankBilinearFusion, self).__init__()
        
        # 1. Projection Layers
        # Project distinct input spaces to a shared 'rank' space
        self.text_proj = nn.Linear(text_dim, rank)
        self.audio_proj = nn.Linear(audio_dim, rank)
        
        # Optional: Add activation or Dropout here if overfitting occurs
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        # 2. Classifier
        # Takes the fused vector (size 'rank') and maps to output
        self.classifier = nn.Linear(rank, output_dim)

    def forward(self, text, audio):
        """
        Args:
            text: Tensor of shape (Batch_Size, text_dim)
            audio: Tensor of shape (Batch_Size, audio_dim)
        """
        # 
        
        # 1. Project to shared rank space
        t = self.text_proj(text) # Shape: (Batch, rank)
        a = self.audio_proj(audio) # Shape: (Batch, rank)
        
        # (Optional) Apply activation
        t = self.relu(t)
        a = self.relu(a)
        
        # 2. Bilinear Fusion (Hadamard Product)
        # Element-wise multiplication approximates the outer product interaction
        # without the massive parameter cost of a full bilinear interaction.
        fused = t * a  # Shape: (Batch, rank)
        
        # 3. Dropout and Classify
        fused = self.dropout(fused)
        logits = self.classifier(fused) # Shape: (Batch, output_dim)
        
        return logits

# --- Example of running the model ---
if __name__ == "__main__":

    concat_features_large()  # Ensure the concatenated CSV is created
    """

    # 1. Setup Data
    dataset = MultimodalDataset("data/processed/concat_early_fusion.csv")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Get a sample batch to check dimensions
    sample_batch = next(iter(loader))
    text_dim = sample_batch['text'].shape[1]  # Should be 768
    audio_dim = sample_batch['audio'].shape[1] # e.g. 768 or 1024
    
    print(f"Detected Input Dims -> Text: {text_dim}, Audio: {audio_dim}")

    # 2. Initialize Model
    model = LowRankBilinearFusion(text_dim=text_dim, audio_dim=audio_dim, rank=64)
    
    # 3. Forward Pass
    logits = model(sample_batch['text'], sample_batch['audio'])
    
    print("Output shape:", logits.shape) # Should be (32, 1)

    """