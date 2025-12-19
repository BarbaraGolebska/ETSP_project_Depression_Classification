import torch
import torch.nn as nn

class LowRankBilinearFusion(nn.Module):
    def __init__(self, text_dim, audio_dim, rank=128):
        super().__init__()
        # 1. Project inputs to a shared 'rank' space
        self.text_proj = nn.Linear(text_dim, rank)
        self.audio_proj = nn.Linear(audio_dim, rank)
        
        # 2. Classifier
        self.classifier = nn.Linear(rank, 1)

    def forward(self, text, audio):
        # Project both to 'rank' size
        t = self.text_proj(text) # [Batch, rank]
        a = self.audio_proj(audio) # [Batch, rank]
        
        # Multiply (Hadamard) in this lower rank space
        fused = t * a 
        
        return self.classifier(fused)
    
import pandas as pd
import numpy as np

def prepare_multimodal_dataframe():
    print("Loading datasets...")
    # 1. Load Text Embeddings (Contains: participant_id, text_feat0...text_feat767, targets)
    df_text = pd.read_csv("data/processed/text_embeddings_aggregated.csv")
    
    # 2. Load Audio Embeddings (Contains: participant_id, and likely hubert features)
    df_audio = pd.read_csv("data/processed/hubert_aggregated_embeddings.csv")

    # --- RENAME COLUMNS TO ENSURE UNIQUENESS ---
    # We assume df_audio might have columns like "0", "1", "feature_0" etc. 
    # We force them to be named "hubert_feat_0", "hubert_feat_1" etc.
    
    # Identify which columns are features (exclude ID and metadata)
    # Adjust this list based on what your hubert CSV actually looks like!
    metadata_cols = ["participant_id", "target_depr", "target_ptsd", "split", "Participant", "id"]
    audio_feat_cols = [c for c in df_audio.columns if c not in metadata_cols]
    
    # Create a renaming map
    rename_map = {old_name: f"hubert_feat_{i}" for i, old_name in enumerate(audio_feat_cols)}
    df_audio = df_audio.rename(columns=rename_map)

    # --- MERGE ---
    print("Merging datasets...")
    # Merge on participant_id and split/targets if they exist in both
    # We only keep rows where we have BOTH text and audio
    df_merged = pd.merge(
        df_text, 
        df_audio[["participant_id"] + list(rename_map.values())], # Only take ID and Features from audio
        on="participant_id", 
        how="inner"
    )

    print(f"Merged Shape: {df_merged.shape}")
    
    # Check if we have targets (assuming they came from df_text)
    if "target_depr" not in df_merged.columns:
        print("Warning: target_depr not found. Make sure it was in df_text.")

    return df_merged


import torch
from torch.utils.data import Dataset, DataLoader

class DepressionDataset(Dataset):
    def __init__(self, df):
        self.df = df
        
        # 1. Identify Text Columns automatically
        self.text_cols = [c for c in df.columns if "text_feat" in c]
        
        # 2. Identify Audio Columns automatically
        self.audio_cols = [c for c in df.columns if "hubert_feat" in c]
        
        print(f"Initialized Dataset with {len(self.text_cols)} Text features and {len(self.audio_cols)} Audio features.")

        # Extract Targets
        self.y = df["target_depr"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Select row
        row = self.df.iloc[idx]
        
        # Get separated features as numpy arrays -> Convert to Tensor
        text_data = torch.tensor(row[self.text_cols].values.astype(np.float32))
        audio_data = torch.tensor(row[self.audio_cols].values.astype(np.float32))
        label = torch.tensor(row["target_depr"], dtype=torch.float32)
        
        return text_data, audio_data, label
    

if __name__ == "__main__":
    # 1. Prepare Data
    df_final = prepare_multimodal_dataframe()
    
    # 2. Create Dataset and Loader
    dataset = DepressionDataset(df_final)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 3. Initialize Model
    # dynamically get dimensions from the dataset
    text_input_dim = len(dataset.text_cols) 
    audio_input_dim = len(dataset.audio_cols)
    
    model = LowRankBilinearFusion(text_dim=text_input_dim, audio_dim=audio_input_dim, rank=64)
    
    # 4. Test a single forward pass
    print("\n--- Testing Model Forward Pass ---")
    for text_batch, audio_batch, label_batch in dataloader:
        output = model(text_batch, audio_batch)
        print("Input Text Shape:", text_batch.shape)
        print("Input Audio Shape:", audio_batch.shape)
        print("Output Shape:", output.shape)

        # 4. SAVE TO CSV
        output_path = "data/processed/lowrank_hubert_text.csv"
        df = pd.DataFrame(output.detach().cpu().numpy())  # convert tensor → numpy → DataFrame
        df.to_csv(output_path, index=False)
        print(f"Saved merged dataframe to {output_path}")
        break # Just run once