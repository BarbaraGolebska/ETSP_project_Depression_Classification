import torch
import torch.nn as nn
import optuna
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import project_utils as utils

# =========================
# PYTORCH HELPERS
# =========================
def create_model(input_dim, device):
    """Initialize a simple linear model with Xavier weights."""
    torch.manual_seed(1)
    model = nn.Linear(input_dim, 1).to(device)
    nn.init.xavier_uniform_(model.weight)
    if model.bias is not None:
        nn.init.zeros_(model.bias)
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
# OPTUNA OBJECTIVE
# =========================
def objective(trial, device, X_np, y_np, groups, kf, oversampler_name):
    # 1. Suggest Hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    epochs = trial.suggest_int("epochs", 10, 40, step=10)


    fold_scores = []

    # 2. Cross-Validation with GroupKFold
    for train_idx, val_idx in kf.split(X_np, y_np, groups=groups):
        # Split (Keep as Numpy for Imblearn)
        X_train_fold, y_train_fold = X_np[train_idx], y_np[train_idx]
        X_val_fold, y_val_fold = X_np[val_idx], y_np[val_idx]

        # 3. Apply Oversampling (Inside fold to prevent leakage)
        sampler = utils.get_oversampler(oversampler_name)
        if sampler:
            X_train_fold, y_train_fold = sampler.fit_resample(X_train_fold, y_train_fold)

        # 4. Convert to Tensors
        X_train_t = torch.tensor(X_train_fold, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train_fold, dtype=torch.float32).to(device)
        X_val_t = torch.tensor(X_val_fold, dtype=torch.float32).to(device)
        
        # 5. Setup Model
        model = create_model(X_train_t.shape[1], device)
        optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        # 6. Train Loop
        for _ in range(epochs):
            train_epoch(model, optimizer, criterion, X_train_t, y_train_t)

        # 7. Evaluate Fold
        model.eval()
        with torch.no_grad():
            logits = model(X_val_t).squeeze()
            y_proba = torch.sigmoid(logits).cpu().numpy()
            if y_proba.ndim == 0: y_proba = np.array([y_proba])
            
            # Optimize for AUC to match the notebook strategy
            auc = roc_auc_score(y_val_fold, y_proba)
        
        fold_scores.append(auc)

    return np.mean(fold_scores)

# =========================
# MAIN FUNCTION
# =========================
ftypes = {
    #"expert_k": "ExpertK_aggregated_features.csv",
    #"bow": "BoW_aggregated_features.csv",
    #"deep_rep": "DeepR_aggregated_features.csv",
    "hubert": "hubert_aggregated_embeddings.csv",
    #"all": "merged_all_features.csv",
    #"all_incl_hubert": "merged_all_features_hubert.csv"
    #"ek_egemaps":"ek_egemaps_aggregated_features.csv",
    #"ek_mfcc":"ek_mfcc_aggregated_features.csv"
}

oversampling_methods = ["None", "RandomOverSampler", "SMOTE", "BorderlineSMOTE"]
#oversampling_methods = ["None"]


def main():
    utils.set_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for ftype, path in ftypes.items():
        # Load Data (Numpy format)
        X_train_np, y_train_np, X_dev_np, y_dev_np, df_train, df_dev = utils.load_processed_data(path)
        
        # Extract groups for proper cross-validation
        groups = df_train["participant_id"].values

        for oversampler_name in oversampling_methods:
            print(f"\n========== BASELINE | {ftype.upper()} | {oversampler_name} ==========")

            # 1. Optuna Study with GroupKFold
            kf = GroupKFold(n_splits=5)
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(
                lambda t: objective(t, device, X_train_np, y_train_np, groups, kf, oversampler_name),
                n_trials=20
            )

            print(f"Best AUC: {study.best_value:.4f} | Params: {study.best_params}")

            # 2. Retrain Final Model on Full Train Set
            
            # A. Apply Oversampling to full training set
            sampler = utils.get_oversampler(oversampler_name)
            if sampler:
                X_train_os, y_train_os = sampler.fit_resample(X_train_np, y_train_np)
            else:
                X_train_os, y_train_os = X_train_np, y_train_np

            # B. Convert to Tensor
            X_train_t = torch.tensor(X_train_os, dtype=torch.float32).to(device)
            y_train_t = torch.tensor(y_train_os, dtype=torch.float32).to(device)

            # C. Initialize & Train
            best = study.best_params
            final_model = create_model(X_train_t.shape[1], device)
            optimizer = getattr(torch.optim, best["optimizer"])(final_model.parameters(), lr=best["lr"])
            criterion = nn.BCEWithLogitsLoss()

            for _ in range(best["epochs"]):
                train_epoch(final_model, optimizer, criterion, X_train_t, y_train_t)

            # 3. Evaluate on Dev Set
            utils.evaluate_and_report(
                model=final_model, 
                X_dev=X_dev_np, 
                y_dev=y_dev_np, 
                df_dev_ids=df_dev, 
                best_params=best,
                model_name="Baseline_Linear",      
                data=ftype,
                oversampler=oversampler_name,
                save_path="baseline_results.csv"
            )

def train_best_model():

    utils.set_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for ftype, path in ftypes.items():
        # Load Data (Numpy format)
        X_train_np, y_train_np, X_dev_np, y_dev_np, df_train, df_dev = utils.load_processed_data(path)
        X_test, y_test, df_test = utils.load_test_data(path)
        
        # Extract groups for proper cross-validation
        groups = df_train["participant_id"].values

        for oversampler_name in oversampling_methods:
            print(f"\n========== BASELINE | {ftype.upper()} | {oversampler_name} ==========")

            # 1. Optuna Study with GroupKFold

            # 2. Retrain Final Model on Full Train Set
            
            # A. Apply Oversampling to full training set
            sampler = utils.get_oversampler(oversampler_name)
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
            criterion = nn.BCEWithLogitsLoss()

            for _ in range(best["epochs"]):
                train_epoch(final_model, optimizer, criterion, X_train_t, y_train_t)

            # D. SAVE THE MODEL (.pkl)
            model_path = f"audio_based_classifier/models/{ftype}_{oversampler_name}_baseline.pkl"
            torch.save(final_model.state_dict(), model_path)
            print(f"Saved model to {model_path}")

            # 3. Evaluate on Dev Set
            utils.evaluate_and_report(
                model=final_model, 
                X_dev=X_test, 
                y_dev=y_test, 
                df_dev_ids=df_test, 
                best_params=best,
                model_name="Baseline_Linear",      
                data=ftype,
                oversampler=oversampler_name,
                save_path="test_baseline_results.csv"
            )


if __name__ == "__main__":
    main()