import torch
import torch.nn as nn
import optuna
import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
try:
    import project_utils as utils
except ImportError:
    from . import project_utils as utils


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
    "expert_k": "ExpertK_aggregated_features.csv",
    "bow": "BoW_aggregated_features.csv",
    "deep_rep": "DeepR_aggregated_features.csv",
    "hubert": "hubert_aggregated_embeddings.csv",
    "BOW_MFCC": "BOWMFCC_aggregated_features.csv",
    "BOW_Egemaps": "BOWEgemaps_aggregated_features.csv",
    "VGG16": "VGG16_aggregated_features.csv",
    "DenseNet201": "DenseNet201_aggregated_features.csv",
    "all": "merged_all_features.csv",
    "all_incl_hubert": "merged_all_features_hubert.csv",
    "ek_egemaps":"ek_egemaps_aggregated_features.csv",
    "ek_mfcc":"ek_mfcc_aggregated_features.csv"
}

#oversampling_methods = ["None", "RandomOverSampler", "SMOTE", "BorderlineSMOTE"]
oversampling_methods = ["ADASYN"]


def train_models(ftypes=ftypes, oversampling_methods=oversampling_methods, save_path="individual_baseline_results.csv"):
    utils.set_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for ftype, path in ftypes.items():
        # Load Data (Numpy format)
        X_train_np, y_train_np, X_dev_np, y_dev_np, df_train, df_dev = utils.load_processed_data(path,"./data/processed/" )
        
        # Extract groups for proper cross-validation
        groups = df_train["participant_id"].values

        for oversampler_name in oversampling_methods:
            print(f"\n========== BASELINE | {ftype.upper()} | {oversampler_name} ==========")

            # 1. Optuna Study with GroupKFold
            kf = GroupKFold(n_splits=5)
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(
                lambda t: objective(t, device, X_train_np, y_train_np, groups, kf, oversampler_name),
                n_trials=500
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

            #save_path="BORRAR_results.csv"

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
                save_path=save_path
            )


def train_best_model(ftypes, oversampling_methods, best=None, save_path="test_baseline_results.csv"):

    utils.set_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for ftype, path in ftypes.items():
        # Load Data (Numpy format)
        X_train_np, y_train_np, X_dev_np, y_dev_np, df_train, df_dev = utils.load_processed_data(path, "./data/processed/")
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
                X_dev=X_dev_np, 
                y_dev=y_dev_np, 
                df_dev_ids=df_dev, 
                best_params=best,
                model_name="Baseline_Linear",      
                data=ftype,
                oversampler=oversampler_name,
                save_path="test_baseline_results.csv"
            )

def evaluate_baseline_single_test(
    data_key,
    ftypes_mapping,
    dev_threshold=0.7539
):
    """
    Train and test a single Baseline HuBERT model configuration
    using fixed hyperparameters (for debugging / validation).
    """

    utils.set_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Fixed hyperparameters (from your row)
    best_params = {
        "lr": 0.09540141357202427,
        "optimizer": "Adam",
        "epochs": 20
    }
    oversampler_name = "ADASYN"

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    if data_key not in ftypes_mapping:
        raise ValueError(f"Data key '{data_key}' not found in ftypes mapping")

    path = ftypes_mapping[data_key]

    X_train_np, y_train_np, _, _, _, _ = utils.load_processed_data(path, "./data/processed/")
    X_test_np, y_test_np, _ = utils.load_test_data(path, "./data/processed/")

    # ------------------------------------------------------------------
    # Oversampling (TRAIN only)
    # ------------------------------------------------------------------
    sampler = utils.get_oversampler(oversampler_name)
    if sampler is not None:
        X_train_np, y_train_np = sampler.fit_resample(X_train_np, y_train_np)

    # ------------------------------------------------------------------
    # Convert to tensors
    # ------------------------------------------------------------------
    X_train_t = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train_np, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test_np, dtype=torch.float32).to(device)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = create_model(X_train_t.shape[1], device)

    optimizer_cls = getattr(torch.optim, best_params["optimizer"])
    optimizer = optimizer_cls(model.parameters(), lr=best_params["lr"])
    criterion = nn.BCEWithLogitsLoss()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    model.train()
    for epoch in range(best_params["epochs"]):
        train_epoch(model, optimizer, criterion, X_train_t, y_train_t)

    # ------------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t).squeeze()
        y_proba = torch.sigmoid(logits).cpu().numpy()

        if y_proba.ndim == 0:
            y_proba = np.array([y_proba])

        # AUC
        try:
            test_auc = roc_auc_score(y_test_np, y_proba)
        except ValueError:
            test_auc = 0.5

        # Apply DEV threshold
        y_pred = (y_proba >= dev_threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test_np, y_pred).ravel()

        precision = precision_score(y_test_np, y_pred, zero_division=0)
        recall = recall_score(y_test_np, y_pred, zero_division=0)
        f1 = f1_score(y_test_np, y_pred, zero_division=0)

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        test_youden = recall + specificity - 1

    # ------------------------------------------------------------------
    # Print results (no CSV)
    # ------------------------------------------------------------------
    print("\n===== TEST RESULTS =====")
    print(f"AUC        : {test_auc:.4f}")
    print(f"Youden     : {test_youden:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1         : {f1:.4f}")
    print(f"TN FP FN TP: {tn} {fp} {fn} {tp}")



def evaluate_on_test_set(results_csv_path, ftypes_mapping):
    """
    Reads existing results CSV, retrains models using best_params, 
    evaluates on the TEST set, and updates the CSV with test metrics.
    """
    utils.set_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the Results CSV
    try:
        df_results = pd.read_csv(results_csv_path)
    except FileNotFoundError:
        print(f"Error: File {results_csv_path} not found.")
        return

    # 2. Define New Columns
    new_cols = [
        'test_auc', 'test_youden', 'test_precision', 'test_recall', 'test_f1',
        'test_TN', 'test_FP', 'test_FN', 'test_TP'
    ]
    
    # Initialize columns if they don't exist
    for col in new_cols:
        if col not in df_results.columns:
            df_results[col] = None

    print(f"Processing {len(df_results)} rows from {results_csv_path}...")

    # 3. Iterate through each row in the CSV
    for index, row in df_results.iterrows():
        data_key = row['Data']
        oversampler_name = row['Oversampler']
        
        # Parse the string representation of the dictionary back into a dict
        
        try:
            best_params = ast.literal_eval(row['best_params'])
            # Ensure Best_Threshold is treated as a float
            dev_threshold = float(row['Best_Threshold'])
        except (ValueError, SyntaxError) as e:
            print(f"Skipping row {index}: Could not parse params. Error: {e}")
            continue

        # Check if we have the file mapping
        if data_key not in ftypes_mapping:
            print(f"Skipping row {index}: Data key '{data_key}' not found in ftypes mapping.")
            continue

        print(f"\n[{index+1}/{len(df_results)}] Processing: {data_key} | {oversampler_name}")
        
        # 4. Load Data
        path = ftypes_mapping[data_key]
        X_train_np, y_train_np, _, _, _, _ = utils.load_processed_data(path, "./data/processed/")
        X_test_np, y_test_np, _ = utils.load_test_data(path, "./data/processed/") # Ensure this function exists in utils

        # 5. Oversampling (Apply to Train only)
        sampler = utils.get_oversampler(oversampler_name)
        if sampler and oversampler_name != "None":
            X_train_os, y_train_os = sampler.fit_resample(X_train_np, y_train_np)
        else:
            X_train_os, y_train_os = X_train_np, y_train_np

        # 6. Convert to Tensors
        X_train_t = torch.tensor(X_train_os, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train_os, dtype=torch.float32).to(device)
        X_test_t = torch.tensor(X_test_np, dtype=torch.float32).to(device)

        # 7. Train Model
        model = create_model(X_train_t.shape[1], device)
        optimizer = getattr(torch.optim, best_params["optimizer"])(model.parameters(), lr=best_params["lr"])
        criterion = nn.BCEWithLogitsLoss()

        model.train()
        for _ in range(best_params["epochs"]):
            train_epoch(model, optimizer, criterion, X_train_t, y_train_t)

        # 8. Test Evaluation
        model.eval()
        with torch.no_grad():
            logits = model(X_test_t).squeeze()
            y_proba = torch.sigmoid(logits).cpu().numpy()
            
            # Handle edge case of single sample
            if y_proba.ndim == 0: y_proba = np.array([y_proba])

            # A. Calculate AUC
            try:
                test_auc = roc_auc_score(y_test_np, y_proba)
            except ValueError:
                test_auc = 0.0 # Handle cases with only one class in test set

            # B. Apply Threshold (CRITICAL: Use the threshold tuned on Dev set)
            y_pred = (y_proba >= dev_threshold).astype(int)

            # C. Confusion Matrix & Derived Metrics
            tn, fp, fn, tp = confusion_matrix(y_test_np, y_pred).ravel()
            
            precision = precision_score(y_test_np, y_pred, zero_division=0)
            recall = recall_score(y_test_np, y_pred, zero_division=0) # Same as Sensitivity
            f1 = f1_score(y_test_np, y_pred, zero_division=0)
            
            # Specificity = TN / (TN + FP)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Youden on Test Set using the Dev Threshold
            test_youden = recall + specificity - 1

        # 9. Update DataFrame
        df_results.at[index, 'test_auc'] = round(test_auc, 4)
        df_results.at[index, 'test_youden'] = round(test_youden, 4)
        df_results.at[index, 'test_precision'] = round(precision, 4)
        df_results.at[index, 'test_recall'] = round(recall, 4)
        df_results.at[index, 'test_f1'] = round(f1, 4)
        df_results.at[index, 'test_TN'] = tn
        df_results.at[index, 'test_FP'] = fp
        df_results.at[index, 'test_FN'] = fn
        df_results.at[index, 'test_TP'] = tp

        # Save periodically in case of crash
        df_results.to_csv(results_csv_path, index=False)
    
    print(f"\nDone! Updated results saved to {results_csv_path}")


if __name__ == "__main__":
    #main()
    #train_models()
    # =========================
    # MAIN EXECUTION
    # =========================
    # Define your file mappings here so the code knows what "concat_hubert_text" maps to

    ftypes = {
    "expert_k": "ExpertK_aggregated_features.csv",
    "bow": "BoW_aggregated_features.csv",
    "deep_rep": "DeepR_aggregated_features.csv",
    "hubert": "hubert_aggregated_embeddings.csv",
    "BOW_MFCC": "BOWMFCC_aggregated_features.csv",
    "BOW_Egemaps": "BOWEgemaps_aggregated_features.csv",
    "VGG16": "VGG16_aggregated_features.csv",
    "DenseNet201": "DenseNet201_aggregated_features.csv",
    "all": "merged_all_features.csv",
    "all_incl_hubert": "merged_all_features_hubert.csv",
    "ek_egemaps":"ek_egemaps_aggregated_features.csv",
    "ek_mfcc":"ek_mfcc_aggregated_features.csv",

    #"concat_hubert_text": "EF_hubert_text.csv", #text + hubert
    #"concat_hubert_expertk_text": "EF_hubert_text_expertk.csv" #text + hubert + expertk (opensmile mfcc and egemaps)
    }
    
    csv_path = "combined_baseline_results.csv" # The file you pasted
    evaluate_on_test_set(csv_path, ftypes)

    #evaluate_baseline_hubert_single_test(data_key="hubert", ftypes_mapping=ftypes)