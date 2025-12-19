import optuna
import pandas as pd
import numpy as np

import ast
import joblib
import os

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
#from catboost import CatBoostClassifier # Added CatBoost

import project_utils as utils

# =========================
# HYPERPARAMETER SPACES
# =========================
def suggest_params(trial, model_type):
    if model_type == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "n_jobs": -1, "random_state": 42
        }
    elif model_type == "xgb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "eval_metric": "logloss", "n_jobs": -1, "random_state": 42,

            "tree_method":"hist", "device":"cuda"

        }
    elif model_type == "lgbm":
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "num_leaves": trial.suggest_int("num_leaves", 4, 32),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "random_state": 42, "n_jobs": -1, "verbosity": -1, # verbosity=-1 is silence for LGBM

            "device": "gpu",
            "gpu_platform_id": 0,
            "gpu_device_id" : 0            
        }
    elif model_type == "cat":
        return {
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "border_count": trial.suggest_int("border_count", 5, 25),
            "verbose": 0, # Use verbose=0 for silence during Optuna evaluation
            "random_state": 42,
            
            "task_type":"GPU",
            "devices":'0'
        }


def get_model_class(model_type):
    return {
        "rf": RandomForestClassifier,
        "xgb": XGBClassifier
    }[model_type]

# =========================
# OBJECTIVE FUNCTION
# =========================
def objective(trial, X, y, groups, kf, oversampler_name, model_type):
    params = suggest_params(trial, model_type)
    ModelClass = get_model_class(model_type)
    
    fold_scores = []
    
    for train_idx, val_idx in kf.split(X, y, groups=groups):
        X_train_fold, y_train_fold = X[train_idx], y[train_idx]
        X_val_fold, y_val_fold = X[val_idx], y[val_idx]

        # Apply Oversampling (Only on Train fold)
        sampler = utils.get_oversampler(oversampler_name)
        if sampler:
            X_train_fold, y_train_fold = sampler.fit_resample(X_train_fold, y_train_fold)

        # Initialize Base Model
        base_model = ModelClass(**params)
        
        # Wrap with Calibration
        model = CalibratedClassifierCV(base_model, cv=3, method='isotonic')
        model.fit(X_train_fold, y_train_fold)
        
        # Evaluate on Validation Fold
        y_proba = model.predict_proba(X_val_fold)[:, 1]
        
        # Optimize for AUC
        auc = roc_auc_score(y_val_fold, y_proba)
        fold_scores.append(auc)

    return np.mean(fold_scores)

# =========================
# MAIN
# =========================
ftypes = {
    "expert_k": "ExpertK_aggregated_features.csv",
    "bow": "BoW_aggregated_features.csv",
    "deep_rep": "DeepR_aggregated_features.csv",
    "hubert": "hubert_aggregated_embeddings.csv",
    "all": "merged_all_features.csv",
    "all_incl_hubert": "merged_all_features_hubert.csv",
    "ek_egemaps":"ek_egemaps_aggregated_features.csv",
    "ek_mfcc":"ek_mfcc_aggregated_features.csv"
}

oversampling_methods = ["None","RandomOverSampler", "SMOTE", "BorderlineSMOTE"]
#oversampling_methods = ["None"]
#MODELS_TO_RUN = ["rf", "xgb", "lgbm", "cat"] 
MODELS_TO_RUN = ["rf","xgb"] 

def main():
    utils.set_seed(1)

    for model_type in MODELS_TO_RUN:
        for ftype, path in ftypes.items():
            # Load Data
            X_train, y_train, X_dev, y_dev, df_train, df_dev = utils.load_processed_data(path)
            
            # Extract Groups for GroupKFold
            groups = df_train["participant_id"].values

            for oversampler_name in oversampling_methods:
                print(f"\n=== {model_type.upper()} | {ftype} | {oversampler_name} ===")

                # 1. Optuna Search with GroupKFold
                kf = GroupKFold(n_splits=5)
                
                study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
                study.optimize(lambda t: objective(t, X_train, y_train, groups, kf, oversampler_name, model_type), n_trials=20)

                print(f"Best AUC: {study.best_value:.4f} with params {study.best_params}")

                # 2. Prepare Final Model
                ModelClass = get_model_class(model_type)
                final_params = study.best_params
                final_params.update({"random_state": 42})
                
                # Set model-specific parameters
                if model_type == "xgb": 
                    final_params.update({"eval_metric": "logloss", "n_jobs": -1})
                elif model_type != "cat": 
                    final_params.update({"n_jobs": -1})
                
                # Oversample Full Training Set
                sampler = utils.get_oversampler(oversampler_name)
                if sampler:
                    X_train_os, y_train_os = sampler.fit_resample(X_train, y_train)
                else:
                    X_train_os, y_train_os = X_train, y_train

                # 3. Feature Importance Check (Base Model Only)
                
                # --- FIX: Ensure only one verbosity parameter is passed to CatBoost ---
                if model_type == "cat":
                    # Pass verbose=0 explicitly to silence the fit operation
                    base_model_for_importance = ModelClass(**final_params, verbose=0)
                else:
                    base_model_for_importance = ModelClass(**final_params)
                    
                base_model_for_importance.fit(X_train_os, y_train_os)
                
                if hasattr(base_model_for_importance, "feature_importances_") or model_type == "cat":
                    
                    if model_type == "cat":
                         # CatBoost uses a different method for importance
                         importance_values = base_model_for_importance.get_feature_importance()
                    else:
                         importance_values = base_model_for_importance.feature_importances_
                         
                    importances = pd.DataFrame({
                        'Feature': df_train.drop(columns=["participant_id", "target_depr", "target_ptsd", "split"]).columns,
                        'Importance': importance_values
                    }).sort_values(by='Importance', ascending=False).head(10)
                    print("\nTop Features (Base Model):\n", importances)

                # 4. Train Calibrated Final Model
                # --- FIX: Ensure the base model for calibration is also silenced if CatBoost ---
                if model_type == "cat":
                    base_model = ModelClass(**final_params, verbose=0) 
                else:
                    base_model = ModelClass(**final_params)

                final_model = CalibratedClassifierCV(base_model, cv=5, method='isotonic')
                final_model.fit(X_train_os, y_train_os)

                # 5. Evaluate using Utility (Calculates AUC & Youden)
                utils.evaluate_and_report(
                    final_model, X_dev, y_dev, df_dev, 
                    final_params, model_type, ftype, oversampler_name, 
                    save_path="tree_results.csv"
                ) 




if __name__ == "__main__":
    main() #to run hyperparameter tuning and evaluation