import pandas as pd
import numpy as np
import random
import os
import csv
import torch
import ast
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

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

# =========================
# DATA LOADING
# =========================
def load_processed_data(filename, base_path="../data/processed/"):
    df = pd.read_csv(f"{base_path}{filename}")
    df_train = df[df["split"] == "train"]
    df_dev = df[df["split"] == "dev"]

    drop_cols = ["participant_id", "target_depr", "target_ptsd", "split"]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train.drop(columns=drop_cols))
    y_train = df_train["target_depr"].values

    X_dev = scaler.transform(df_dev.drop(columns=drop_cols))
    y_dev = df_dev["target_depr"].values
    
    return X_train, y_train, X_dev, y_dev, df_train, df_dev

def load_test_data(filename, base_path="../data/processed/"):
    df = pd.read_csv(f"{base_path}{filename}")
    df_train = df[df["split"] == "train"]
    df_test = df[df["split"] == "test"]   # NEW

    drop_cols = ["participant_id", "target_depr", "target_ptsd", "split"]
    
    scaler = StandardScaler()
    
    # Fit only on train
    X_train = scaler.fit_transform(df_train.drop(columns=drop_cols))
    y_train = df_train["target_depr"].values


    X_test = scaler.transform(df_test.drop(columns=drop_cols))
    y_test = df_test["target_depr"].values

    return (X_test, y_test, df_test)


# =========================
# SAMPLING FACTORY
# =========================
def get_sampler(name, seed=42):
    if name == "RandomOverSampler":
        return RandomOverSampler(random_state=seed)
    elif name == "SMOTE":
        return SMOTE(random_state=seed)
    elif name == "BorderlineSMOTE":
        return BorderlineSMOTE(random_state=seed)
    elif name == "RandomUnderSampler":
        return RandomUnderSampler(random_state=seed)
    elif name == "ADASYN":
        return ADASYN(random_state=seed)
    return None


# =========================
# DIMENSIONALITY REDUCTION
# =========================
def dim_reduction(X_train, y_train, X_dev, method = "PCA", n_components=50):
    """Applies dimensionality reduction to the feature matrix X."""
    if method == "PCA":
        n_components = min(n_components, X_train.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        X_train_red = pca.fit_transform(X_train)
        X_dev_red   = pca.transform(X_dev)
    elif method == "SelectKBest":
        k = min(n_components, X_train.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_red = selector.fit_transform(X_train, y_train)
        X_dev_red   = selector.transform(X_dev)
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")
    return X_train_red, X_dev_red


# =========================
# METRICS
# =========================
def find_best_youden(y_true, y_proba):
    """
    Finds optimal threshold maximizing Youden Index (J).
    J = Sensitivity + Specificity - 1
    """
    thresholds = np.unique(y_proba)
    best_youden = -1
    best_threshold = 0.5

    for thr in thresholds:
        if thr < 0.01 or thr > 0.99:
            continue
        y_pred = (y_proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        sensitivity = tp / (tp + fn + 1e-12)
        specificity = tn / (tn + fp + 1e-12)
        youden = sensitivity + specificity - 1

        if youden > best_youden:
            best_youden = youden
            best_threshold = thr

    return best_youden, best_threshold

def compute_youden(y_true, y_proba, threshold):
    """
    Computes Youden Index for a given threshold.
    """
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    youden = sensitivity + specificity - 1

    return youden


# =========================
# EVALUATION & REPORTING
# =========================
def evaluate_and_report(model, 
                        X_dev, 
                        y_dev, 
                        df_dev_ids, 
                        best_params, 
                        model_name, 
                        data, 
                        sampler, 
                        reduction_method, 
                        save_path="experiment_results.csv"):
    """
    Predicts probabilities, calculates AUC, finds best Youden threshold, 
    prints report, and saves to CSV.
    """
    # 1. Get Probabilities
    if isinstance(model, torch.nn.Module):
        model.eval()
        with torch.no_grad():
            device = next(model.parameters()).device
            if isinstance(X_dev, np.ndarray):
                X_tensor = torch.tensor(X_dev, dtype=torch.float32).to(device)
            else:
                X_tensor = X_dev.to(device)
            logits = model(X_tensor)
            y_proba = torch.sigmoid(logits).cpu().numpy().flatten()
    else:
        y_proba = model.predict_proba(X_dev)[:, 1]

    # 2. Calculate Metrics
    # AUC-ROC (Ranking metric)
    auc = roc_auc_score(y_dev, y_proba)
    
    # Youden Index (Thresholding metric)
    if "Best_Threshold" in best_params: # use predefined threshold
        best_thr = best_params["Best_Threshold"]
        ydn = compute_youden(y_dev, y_proba, best_thr)
    else:
        ydn, best_thr = find_best_youden(y_dev, y_proba)
    
    # Apply Threshold
    y_pred = (y_proba >= best_thr).astype(int)
    cm = confusion_matrix(y_dev, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # 3. Console Output
    print(f"\n--- Evaluation: {model_name} ---")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Best Youden: {ydn:.4f} at threshold={best_thr:.3f}")
    print(f"Confusion Matrix:\n{cm}")

    # Misdiagnosed Analysis
    fn_mask = (y_dev == 1) & (y_pred == 0)
    misdiagnosed = df_dev_ids.loc[fn_mask, "participant_id"].values
    print("False Negative IDs (Misdiagnosed):", misdiagnosed)
    print("-" * 40)

    # 4. CSV Writing
    file_exists = os.path.isfile(save_path)
    
    try:
        with open(save_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            
            if not file_exists:
                header = [
                    "Model_Name","Data","Sampler", "Reduction_Method", 
                    "AUC", "Youden_Index", "Best_Threshold", 
                    "TN", "FP", "FN", "TP", "False_Negative_IDs", "best_params"
                ]
                writer.writerow(header)
            
            writer.writerow([
                model_name,
                data,
                sampler,
                reduction_method,
                f"{auc:.4f}",
                f"{ydn:.4f}",
                f"{best_thr:.3f}",
                tn, fp, fn, tp,
                str(list(misdiagnosed)),
                str(best_params)
            ])
        print(f"Results successfully appended to {save_path}")
    except Exception as e:
        print(f"Error writing to CSV: {e}")
    
    return auc


# =========================
# HYPERPARAMETER EXTRACTION
# =========================
def get_best_params_by_youden(
    df: pd.DataFrame,
    group_cols=["Model_Name", "Data"],
    model_name: str = "MLP",
):
    """
    Extracts the best hyperparameters for each group based on the highest Youden Index.
    """

    # Filter by model name if provided
    if model_name is not None:
        df = df[df["Model_Name"] == model_name].copy()

    best_configs = {}

    for group_key, group_df in df.groupby(group_cols):
        # group_key = (Model_Name, Data, Sampler, Reduction_Method)
        idx = group_df["Youden_Index"].idxmax()
        best_row = group_df.loc[idx]

        try:
            params = ast.literal_eval(best_row["best_params"])
        except Exception:
            params = {}

        params["Best_Threshold"] = best_row["Best_Threshold"]
        params["Sampler"] = str(best_row["Sampler"])
        params["Reduction_Method"] = str(best_row["Reduction_Method"])

        best_configs[group_key] = params

    return best_configs