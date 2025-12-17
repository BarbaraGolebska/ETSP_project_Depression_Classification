import torch
import torch.nn as nn
import optuna
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import project_utils as utils

optuna.logging.set_verbosity(optuna.logging.WARNING)
NUM_TRIALS = 800

# early stopping parameters
MAX_EPOCHS = 50
PATIENCE = 5

# =========================
# PYTORCH HELPERS
# =========================

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # return logits of shape (batch,)
        return self.net(x).squeeze(-1)

def create_model(input_dim, device, hidden_dim=128, dropout=0.1):
    model = MLPClassifier(input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    # Initialize weights with Xavier initialization
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return model

def train_epoch(model, optimizer, criterion, X_tensor, y_tensor):
    """Performs one epoch of training."""
    model.train()
    optimizer.zero_grad()
    output = model(X_tensor).squeeze()
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()
    return loss

# =========================
# UTILITY FUNCTIONS
# =========================

def get_pos_weights(y, device):
    """Calculate positive class weight for imbalanced binary classification."""
    n_neg = np.sum(y == 0)
    n_pos = np.sum(y == 1)
    pos_weight_value = n_neg / n_pos
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    return pos_weight

def dimention_reduction(X_train, y_train, X_dev, method = "PCA", n_components=50):
    """Applies dimensionality reduction to the feature matrix X."""
    # 1. Standardize Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled   = scaler.transform(X_dev)

    # 2. Apply Dimensionality Reduction
    if method == "PCA":
        n_components = min(n_components, X_train_scaled.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        X_train_red = pca.fit_transform(X_train_scaled)
        X_dev_red   = pca.transform(X_dev_scaled)
    elif method == "SelectKBest":
        k = min(n_components, X_train_scaled.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_red = selector.fit_transform(X_train_scaled, y_train)
        X_dev_red   = selector.transform(X_dev_scaled)
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")
    return X_train_red, X_dev_red

# =========================
# OPTUNA OBJECTIVE
# =========================
def objective(trial, device, X_np, y_np, groups, kf, sampler_name):
    # 1. Suggest Hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    hidden_dim = trial.suggest_int("hidden_dim", 16, 256, step=16)
    dropout = trial.suggest_float("dropout", 0.2, 0.6, step=0.1)

    fold_scores = []

    # 2. Cross-Validation with GroupKFold
    for train_idx, val_idx in kf.split(X_np, y_np, groups=groups):
        # Split (Keep as Numpy for Imblearn)
        X_train_fold, y_train_fold = X_np[train_idx], y_np[train_idx]
        X_val_fold, y_val_fold = X_np[val_idx], y_np[val_idx]

        # 3. Apply Sampling (Inside fold to prevent leakage)
        sampler = utils.get_sampler(sampler_name)
        if sampler:
            X_train_fold, y_train_fold = sampler.fit_resample(X_train_fold, y_train_fold)

        # 4. Convert to Tensors
        X_train_t = torch.tensor(X_train_fold, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train_fold, dtype=torch.float32).to(device)
        X_val_t = torch.tensor(X_val_fold, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val_fold, dtype=torch.float32).to(device)
        
        # 5. Setup Model
        model = create_model(X_train_t.shape[1], device, hidden_dim=hidden_dim, dropout=dropout)
        optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)
        if sampler_name == "None":
            w = get_pos_weights(y_train_fold, device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=w)
        else:
            criterion = nn.BCEWithLogitsLoss()

        # 6. Train Loop with Early Stopping
        #   * best model based on validation AUC
        #   * early stopping based on validation loss
        best_auc_fold = -np.inf
        best_val_loss   = float("inf")
        epochs_no_improve = 0

        for _ in range(MAX_EPOCHS):
            train_epoch(model, optimizer, criterion, X_train_t, y_train_t)

            # 7. Evaluate Fold
            model.eval()
            with torch.no_grad():
                logits = model(X_val_t).squeeze()

                # loss for early stopping
                val_loss = criterion(logits, y_val_t).item()

                # AUC for model selection
                y_proba = torch.sigmoid(logits).cpu().numpy()
                if y_proba.ndim == 0: y_proba = np.array([y_proba])
                auc = roc_auc_score(y_val_fold, y_proba)
            
            # update best model based on AUC
            if auc > best_auc_fold + 1e-4:
                best_auc_fold = auc
            
            # early stopping based on loss
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    break
        
        fold_scores.append(best_auc_fold)

    return float(np.mean(fold_scores))

# =========================
# MAIN FUNCTION
# =========================
ftypes = {

    "concat_hubert_text": "EF_hubert_text.csv",
    "concat_hubert_expertk_text": "EF_hubert_text_expertk.csv"
}

sampling_methods = ["None", "RandomOverSampler", "SMOTE", "BorderlineSMOTE", "RandomUnderSampler"]
dim_reduction_methods = ["PCA", "SelectKBest"]

def main():
    utils.set_seed(1)
    # Select Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    for ftype, path in ftypes.items():
        for dim_method in dim_reduction_methods:
            # Load Data (Numpy format)
            X_train_np, y_train_np, X_dev_np, y_dev_np, df_train, df_dev = utils.load_processed_data(path)

            # Standardize and reduce dimensionality
            X_train_np, X_dev_np = dimention_reduction(X_train_np, y_train_np, X_dev_np, method=dim_method, n_components=50)
            
            # Extract groups for proper cross-validation
            groups = df_train["participant_id"].values

            for sampler_name in sampling_methods:
                print(f"\n========== MLP | {ftype.upper()} | {sampler_name} | {dim_method} ==========")

                # 1. Optuna Study with GroupKFold
                kf = GroupKFold(n_splits=5)
                study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
                study.optimize(
                    lambda t: objective(t, device, X_train_np, y_train_np, groups, kf, sampler_name),
                    n_trials=NUM_TRIALS,
                    show_progress_bar=True
                )

                print(f"Best AUC: {study.best_value:.4f} | Params: {study.best_params}")

                # 2. Retrain Final Model on Full Train Set
                
                # A. Apply Sampling to full training set
                sampler = utils.get_sampler(sampler_name)
                if sampler:
                    X_train_os, y_train_os = sampler.fit_resample(X_train_np, y_train_np)
                else:
                    X_train_os, y_train_os = X_train_np, y_train_np

                # B. Convert to Tensor
                X_train_t = torch.tensor(X_train_os, dtype=torch.float32).to(device)
                y_train_t = torch.tensor(y_train_os, dtype=torch.float32).to(device)
                X_dev_t = torch.tensor(X_dev_np, dtype=torch.float32).to(device)
                y_dev_t = torch.tensor(y_dev_np, dtype=torch.float32).to(device)

                # C. Initialize & Train
                best = study.best_params
                final_model = create_model(X_train_t.shape[1], device, hidden_dim=best["hidden_dim"], dropout=best["dropout"])
                optimizer = getattr(torch.optim, best["optimizer"])(final_model.parameters(), lr=best["lr"], weight_decay=best["weight_decay"])
                if sampler_name == "None":
                    w = get_pos_weights(y_train_os, device)
                    criterion = nn.BCEWithLogitsLoss(pos_weight=w)
                else:
                    criterion = nn.BCEWithLogitsLoss()

                # D. Train for best number of epochs with early stopping
                #   * best model based on validation AUC
                #   * early stopping based on validation loss
                best_auc_dev = -np.inf
                best_val_loss = np.inf
                epochs_no_improve = 0
                best_state = None

                for _ in range(MAX_EPOCHS):
                    train_epoch(final_model, optimizer, criterion, X_train_t, y_train_t) 

                    final_model.eval()
                    with torch.no_grad():
                        logits = final_model(X_dev_t).squeeze()

                        # loss for early stopping
                        val_loss = criterion(logits, y_dev_t).item()

                        # AUC for model selection
                        y_proba = torch.sigmoid(logits).cpu().numpy()
                        if y_proba.ndim == 0:
                            y_proba = np.array([y_proba])
                        auc = roc_auc_score(y_dev_np, y_proba)

                    # update best model based on AUC
                    if auc > best_auc_dev + 1e-4:
                        best_auc_dev = auc
                        best_state = {
                            k: v.detach().cpu().clone()
                            for k, v in final_model.state_dict().items()
                        }
                    
                    # early stopping based on loss
                    if val_loss < best_val_loss - 1e-4:
                        best_val_loss = val_loss
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= PATIENCE:
                            break

                # restore best state before final evaluation
                if best_state is not None:
                    final_model.load_state_dict(best_state)

                # 3. Evaluate on Dev Set
                utils.evaluate_and_report(
                    model=final_model, 
                    X_dev=X_dev_np, 
                    y_dev=y_dev_np, 
                    df_dev_ids=df_dev, 
                    best_params=best,
                    model_name="MLP",      
                    data=ftype,
                    sampler=sampler_name,
                    reduction_method=dim_method,
                    save_path="EF_MLP_dev.csv"
                )

def train_best_model():

    utils.set_seed(1)
    # Select Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    for ftype, path in ftypes.items():
        # Load Data (Numpy format)
        X_train_np, y_train_np, X_dev_np, y_dev_np, df_train, df_dev = utils.load_processed_data(path)
        X_test, y_test, df_test = utils.load_test_data(path)
        
        # Extract groups for proper cross-validation
        groups = df_train["participant_id"].values

        for sampler_name in sampling_methods:
            print(f"\n========== MLP | {ftype.upper()} | {sampler_name} ==========")

            # 1. Optuna Study with GroupKFold

            # 2. Retrain Final Model on Full Train Set
            
            # A. Apply Sampling to full training set
            sampler = utils.get_sampler(sampler_name)
            if sampler:
                X_train_os, y_train_os = sampler.fit_resample(X_train_np, y_train_np)
            else:
                X_train_os, y_train_os = X_train_np, y_train_np

            # B. Convert to Tensor
            X_train_t = torch.tensor(X_train_os, dtype=torch.float32).to(device)
            y_train_t = torch.tensor(y_train_os, dtype=torch.float32).to(device)

            # C. Initialize & Train
            best = {'lr': 0.09840764582498135, 'optimizer': 'Adam', 'epochs': 30}
            final_model = create_model(X_train_t.shape[1], device)
            optimizer = getattr(torch.optim, best["optimizer"])(final_model.parameters(), lr=best["lr"])
            if sampler_name == "None":
                w = get_pos_weights(y_train_os, device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=w)
            else:
                criterion = nn.BCEWithLogitsLoss()

            for _ in range(best["epochs"]):
                train_epoch(final_model, optimizer, criterion, X_train_t, y_train_t)

            # D. SAVE THE MODEL (.pkl)
            model_path = f"audio_based_classifier/models/{ftype}_{sampler_name}_baseline.pkl"
            torch.save(final_model.state_dict(), model_path)
            print(f"Saved model to {model_path}")

            # 3. Evaluate on Dev Set
            utils.evaluate_and_report(
                model=final_model, 
                X_dev=X_test, 
                y_dev=y_test, 
                df_dev_ids=df_test, 
                best_params=best,
                model_name="MLP",      
                data=ftype,
                sampler=sampler_name,
                save_path="EF_MLP_dev.csv"
            )


if __name__ == "__main__":
    main()