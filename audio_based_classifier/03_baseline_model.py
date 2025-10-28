import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.metrics import precision_recall_curve, confusion_matrix

# =========================
# REPRODUCIBILITY
# =========================
def set_seed(seed=1):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(1)

# =========================
# MODEL & TRAINING HELPERS
# =========================
def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

def create_model(input_dim, device):
    torch.manual_seed(1)
    model = nn.Linear(input_dim, 1).to(device)
    init_weights(model)
    return model

def create_optimizer(model, lr, optimizer_type):
    return torch.optim.Adam(model.parameters(), lr=lr) if optimizer_type=="Adam" else \
           torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

def train_model(model, optimizer, scheduler, criterion, X, y):
    model.train()
    optimizer.zero_grad()
    output = model(X).squeeze()
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    scheduler.step(loss)

def compute_best_f2(y_true, y_proba, beta=2):
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    f_scores = (1 + beta**2) * prec * rec / ((beta**2 * prec) + rec + 1e-12)
    best_idx = np.argmax(f_scores[:-1])
    return f_scores[best_idx], thr[best_idx]

# =========================
# GRID SEARCH FUNCTION
# =========================
def grid_search(device, X, y, param_grid, kf):
    best_f2, best_params, best_model, best_threshold = 0, None, None, 0

    for params in ParameterGrid(param_grid):
        print(f"\nTesting parameters: {params}")
        fold_scores = []

        for train_idx, val_idx in kf.split(X.cpu(), y.cpu()):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            model = create_model(X_train.shape[1], device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = create_optimizer(model, params["lr"], params["optimizer_type"])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

            for _ in range(params["epochs"]):
                train_model(model, optimizer, scheduler, criterion, X_train, y_train)

            model.eval()
            with torch.no_grad():
                y_proba = torch.sigmoid(model(X_val)).cpu().numpy().flatten()
                f2, threshold = compute_best_f2(y_val.cpu().numpy(), y_proba)
            fold_scores.append(f2)

        avg_f2 = np.mean(fold_scores)
        print(f"Average F2: {avg_f2:.4f}")

        if avg_f2 > best_f2:
            best_f2 = avg_f2
            best_params = params
            best_model = model
            best_threshold = threshold
            print(f"→ New best F2={best_f2:.4f} at threshold={best_threshold:.3f}")

    return best_model, best_params, best_f2, best_threshold

# =========================
# MAIN FUNCTION
# =========================
ftypes = {
    "bow": "BoW_aggregated_features.csv",
    "deep_rep": "DeepR_aggregated_features.csv",
    "expert_k": "ExpertK_aggregated_features.csv",
    "all": "merged_all_features.csv"
}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler()

    for ftype, path in ftypes.items():
        df = pd.read_csv(f"data/processed/{path}")
        df_train, df_dev = df[df["split"]=="train"], df[df["split"]=="dev"]

        # =========================
        # PREPROCESS FEATURES
        # =========================
        drop_cols = ["participant_id", "target_depr", "target_ptsd", "split"]
        X_train = pd.DataFrame(scaler.fit_transform(df_train.drop(columns=drop_cols)),
                               columns=df_train.drop(columns=drop_cols).columns,
                               index=df_train.index)
        y_train = df_train["target_depr"]

        X_dev = pd.DataFrame(scaler.transform(df_dev.drop(columns=drop_cols)),
                             columns=df_dev.drop(columns=drop_cols).columns,
                             index=df_dev.index)
        y_dev = df_dev["target_depr"]

        # =========================
        # CONVERT TO TENSORS
        # =========================
        X_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)

        # =========================
        # GRID SEARCH
        # =========================
        param_grid = {"lr":[0.01, 0.001], "optimizer_type":["Adam","SGD"], "epochs":[10,20,30,40]}
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)

        best_model, best_params, best_f2, best_threshold = grid_search(device, X_tensor, y_tensor, param_grid, kf)
        print(f"\nBest params for {ftype}: {best_params}, Best F2: {best_f2:.4f} at threshold {best_threshold:.3f}")

        # =========================
        # TRAIN FINAL MODEL ON FULL TRAINING SET
        # =========================
        final_model = create_model(X_tensor.shape[1], device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = create_optimizer(final_model, best_params["lr"], best_params["optimizer_type"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

        for _ in range(best_params["epochs"]):
            train_model(final_model, optimizer, scheduler, criterion, X_tensor, y_tensor)

        # =========================
        # EVALUATE ON DEV SET
        # =========================
        X_dev_tensor = torch.tensor(X_dev.values, dtype=torch.float32).to(device)
        y_dev_tensor = torch.tensor(y_dev.values, dtype=torch.float32).to(device)

        final_model.eval()
        with torch.no_grad():
            y_proba = torch.sigmoid(final_model(X_dev_tensor)).cpu().numpy().flatten()
            f2_dev, best_threshold = compute_best_f2(y_dev.values, y_proba)
        
        y_pred = (y_proba >= best_threshold).astype(int)
        cm = confusion_matrix(y_dev.values, y_pred)

        print(f"Final model F2 on dev set: {f2_dev:.4f} at threshold={best_threshold:.3f}")
        print(f"Confusion Matrix for {ftype}:\n{cm}")

        # Boolean mask for true positives
        true_positive_mask = (y_dev.values == 1) & (y_pred == 0)

        # Get participant IDs for true positives
        participant_ids_tp = df_dev.loc[true_positive_mask, "participant_id"].values

        print("Participant IDs misdiagnosed (False Nagatives):")
        print(participant_ids_tp)

if __name__ == "__main__":
    main()
