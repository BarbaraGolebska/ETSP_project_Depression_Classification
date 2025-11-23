import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import optuna
import joblib
from sklearn.preprocessing import StandardScaler # Import explicitly

# Import project utilities
import project_utils as utils

# =========================
# CONFIG & CONSTANTS
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Update this path to the actual location of your *_densenet201.csv files
SEQUENCE_DATA_PATH = "data/raw/features/" 
MAX_SEQ_LEN = 1000 # Cap for memory management
BATCH_SIZE = 8

# =========================
# DATASET & LOADING
# =========================
class DenseNetSeqDataset(Dataset):
    def __init__(self, participant_ids, labels_map, feature_dir, scaler=None):
        self.pids = list(participant_ids)
        self.labels_map = labels_map
        self.feature_dir = feature_dir
        self.scaler = scaler

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, i):
        pid = int(self.pids[i])
        # Assumes filename format: {pid}_densenet201.csv
        file_path = os.path.join(self.feature_dir, f"{pid}_densenet201.csv")
        
        try:
            df = pd.read_csv(file_path)
            # Filter feature columns (starting with neuron_)
            fcols = [c for c in df.columns if c.startswith("neuron_")]
            X = df[fcols].values.astype(np.float32)
        except FileNotFoundError:
            # Fallback for missing files (or handle gracefully)
            print(f"Warning: File not found for {pid}, returning zeros.")
            X = np.zeros((10, 1920), dtype=np.float32) # Dummy shape

        if self.scaler:
            # Transform the features using the scaler fitted on sequence data
            X = self.scaler.transform(X)

        x_tensor = torch.from_numpy(X).float()
        
        # Get Label
        label = self.labels_map.get(pid, 0) # Default to 0 if missing
        label_tensor = torch.tensor(float(label), dtype=torch.float32)

        return {
            "x": x_tensor,
            "length": x_tensor.shape[0],
            "label": label_tensor,
            "id": pid
        }

def collate_pad(batch):
    # Sort by length for potential packing (optional, skipping here for simplicity)
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    
    # Pad to max length in this batch
    B = len(batch)
    F = batch[0]["x"].shape[1]
    T_max = lengths.max().item()
    
    # Prepare padded tensor [B, T, F]
    x_padded = torch.zeros(B, T_max, F, dtype=torch.float32)
    labels = torch.zeros(B, dtype=torch.float32)
    ids = []

    for i, b in enumerate(batch):
        L = b["length"]
        x_padded[i, :L, :] = b["x"]
        labels[i] = b["label"]
        ids.append(b["id"])

    # Permute for Conv1d: [B, F, T]
    x_padded = x_padded.permute(0, 2, 1).contiguous()
    
    return x_padded, labels, ids

def load_sequence_data(base_path="data/processed/", sequence_dir=SEQUENCE_DATA_PATH):
    """
    Loads splits from project_utils standard processed file to get IDs,
    then constructs Sequence Datasets by fitting scaler on the sequence files.
    """
    print("Loading reference data to determine splits...")
    # Load reference IDs and Labels from the standard aggregated file
    # We use this ONLY to know which IDs belong to Train/Dev splits
    ref_df = pd.read_csv(f"{base_path}DeepR_aggregated_features.csv") 
    
    train_df = ref_df[ref_df["split"] == "train"]
    dev_df = ref_df[ref_df["split"] == "dev"]
    
    # Create Label Maps
    train_labels = dict(zip(train_df["participant_id"], train_df["target_depr"]))
    dev_labels = dict(zip(dev_df["participant_id"], dev_df["target_depr"]))
    
    # --- FIX: Fit Scaler on Sequence Data using Partial Fit ---
    print("Fitting scaler on training sequences (this may take a moment)...")
    scaler = StandardScaler()
    
    train_pids = train_df["participant_id"].values
    for pid in train_pids:
        file_path = os.path.join(sequence_dir, f"{pid}_densenet201.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Filter feature columns
            fcols = [c for c in df.columns if c.startswith("neuron_")]
            X_part = df[fcols].values.astype(np.float32)
            scaler.partial_fit(X_part)
        else:
            print(f"Skipping scaler fit for missing file: {pid}")

    print("Scaler fitting complete.")
    return train_labels, dev_labels, scaler, train_df, dev_df

# =========================
# MODEL ARCHITECTURE
# =========================
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k, p_drop):
        super().__init__()
        pad = (k - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=k, padding=pad),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Conv1d(c_out, c_out, kernel_size=k, padding=pad),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv1d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.proj(x)

class TemporalCNN(nn.Module):
    def __init__(self, in_features, channels=128, kernel_size=5, n_blocks=3, dropout=0.2):
        super().__init__()
        self.bottle = nn.Sequential(
            nn.Conv1d(in_features, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        blocks = []
        for _ in range(n_blocks):
            blocks.append(ConvBlock(channels, channels, kernel_size, p_drop=dropout))
        self.backbone = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),    # Global Average Pooling -> [B, C, 1]
            nn.Flatten(),               # [B, C]
            nn.Dropout(dropout),
            nn.Linear(channels, 1)      # Binary classification output (logit)
        )

    def forward(self, x):
        # x shape: [B, F, T]
        x = self.bottle(x)
        x = self.backbone(x)
        logits = self.head(x)
        return logits 

# =========================
# TRAINING UTILS
# =========================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device).unsqueeze(1) # y shape [B, 1]
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        
    return total_loss / len(loader.dataset)

def validate(model, loader, device):
    model.eval()
    preds = []
    targets = []
    
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds.extend(probs)
            targets.extend(y.numpy())
            
    try:
        auc = utils.roc_auc_score(targets, preds)
    except:
        auc = 0.5
    return auc

# =========================
# OPTUNA OBJECTIVE
# =========================
def objective(trial, train_labels, dev_labels, feature_dir, scaler, input_dim):
    # 1. Suggest Hyperparams
    params = {
        "channels": trial.suggest_categorical("channels", [64, 128, 256]),
        "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),
        "n_blocks": trial.suggest_int("n_blocks", 1, 4),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    }

    # 2. Setup DataLoaders
    train_ds = DenseNetSeqDataset(train_labels.keys(), train_labels, feature_dir, scaler)
    dev_ds = DenseNetSeqDataset(dev_labels.keys(), dev_labels, feature_dir, scaler)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pad, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pad, num_workers=0)

    # 3. Setup Model
    model = TemporalCNN(input_dim, 
                        channels=params["channels"], 
                        kernel_size=params["kernel_size"], 
                        n_blocks=params["n_blocks"], 
                        dropout=params["dropout"]).to(DEVICE)
    
    # Calculate class weights for Imbalanced Data
    y_vals = list(train_labels.values())
    n_pos = sum(y_vals)
    n_neg = len(y_vals) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], device=DEVICE) if n_pos > 0 else torch.tensor([1.0], device=DEVICE)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

    # 4. Train Loop
    best_auc = 0.0
    
    for epoch in range(5): # Short epochs for search
        train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        auc = validate(model, dev_loader, DEVICE)
        
        trial.report(auc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        best_auc = max(best_auc, auc)

    return best_auc

# =========================
# MAIN
# =========================
def main():
    utils.set_seed(1)
    
    print(f"Running on {DEVICE}")
    print("Loading Data Indices and creating Datasets...")
    
    # 1. Load Data Indices from the aggregated CSV and FIT SCALER on Sequence Files
    if not os.path.exists(SEQUENCE_DATA_PATH):
        print(f"[ERROR] Sequence features directory not found: {SEQUENCE_DATA_PATH}")
        print("Please check SEQUENCE_DATA_PATH config.")
        return

    train_labels, dev_labels, scaler, train_df_ref, dev_df_ref = load_sequence_data(sequence_dir=SEQUENCE_DATA_PATH)
    
    # Check input dimension from one file
    if len(train_labels) > 0:
        sample_pid = list(train_labels.keys())[0]
        sample_file = os.path.join(SEQUENCE_DATA_PATH, f"{sample_pid}_densenet201.csv")
        try:
            input_dim = pd.read_csv(sample_file).filter(like="neuron_").shape[1]
        except:
            input_dim = 1920 # Default DenseNet size
    else:
        input_dim = 1920

    print(f"Input Feature Dimension: {input_dim}")
    print(f"Train samples: {len(train_labels)}, Dev samples: {len(dev_labels)}")

    # 2. Optuna Search
    print("\n=== Starting Optuna Search ===")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    
    study.optimize(
        lambda t: objective(t, train_labels, dev_labels, SEQUENCE_DATA_PATH, scaler, input_dim), 
        n_trials=20
    )

    print(f"Best AUC: {study.best_value:.4f}")
    print(f"Best Params: {study.best_params}")

    # 3. Final Training
    best_params = study.best_params
    
    final_model = TemporalCNN(input_dim, 
                        channels=best_params["channels"], 
                        kernel_size=best_params["kernel_size"], 
                        n_blocks=best_params["n_blocks"], 
                        dropout=best_params["dropout"]).to(DEVICE)

    # Re-create loaders
    train_ds = DenseNetSeqDataset(train_labels.keys(), train_labels, SEQUENCE_DATA_PATH, scaler)
    dev_ds = DenseNetSeqDataset(dev_labels.keys(), dev_labels, SEQUENCE_DATA_PATH, scaler)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pad)
    
    # Calculate Weights again for final training
    y_vals = list(train_labels.values())
    n_pos = sum(y_vals)
    n_neg = len(y_vals) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], device=DEVICE) if n_pos > 0 else torch.tensor([1.0], device=DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])

    print("\n=== Training Final Model ===")
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, collate_fn=collate_pad)
    
    for epoch in range(15): # Train longer for final model
        loss = train_one_epoch(final_model, train_loader, criterion, optimizer, DEVICE)
        if epoch % 5 == 0:
            val_auc = validate(final_model, dev_loader, device=DEVICE)
            print(f"Epoch {epoch}: Loss {loss:.4f}, Val AUC {val_auc:.4f}")

    # 4. Evaluation & Reporting via project_utils
    print("\nPreparing Evaluation Tensor...")
    
    # We load the full dev set as one batch to get tensors for the utility function
    # Note: If this is too large for GPU memory, you might need to run evaluate_and_report on CPU
    dev_loader_full = DataLoader(dev_ds, batch_size=len(dev_ds), shuffle=False, collate_fn=collate_pad)
    X_dev_tensor, y_dev_tensor, _ = next(iter(dev_loader_full))
    
    y_dev_numpy = y_dev_tensor.numpy().astype(int)
    
    final_model.eval()
    
    # Pass the result to the project_utils evaluation function
    utils.evaluate_and_report(
        final_model, 
        X_dev_tensor, 
        y_dev_numpy, 
        dev_df_ref, 
        best_params, 
        model_name="TemporalCNN", 
        data="DeepR_Sequences", 
        oversampler="None",
        save_path="cnn_results.csv"
    )

if __name__ == "__main__":
    main()