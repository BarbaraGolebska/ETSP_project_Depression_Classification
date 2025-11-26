import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime
import logging

import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    precision_score, recall_score, f1_score
)
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

import lightgbm as lgb
from catboost import CatBoostClassifier

logging.basicConfig(level=logging.INFO)

##################################################
# ---------------- PREPROCESSING -----------------
##################################################

def remove_constant_cols_local(X_train, X_val):
    std = np.nanstd(X_train, axis=0)
    mask = std > 1e-8
    return X_train[:, mask], X_val[:, mask], mask

def impute_nans(X):
    col_means = np.nanmean(X, axis=0)
    idx = np.isnan(X)
    X[idx] = np.take(col_means, np.where(idx)[1])
    return X

##################################################
# ---------------- MODEL TRAINERS ----------------
##################################################

def train_lightgbm(trial, X_train, y_train, scale_pos_weight):
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 512),
        "max_depth": trial.suggest_int("max_depth", -1, 16),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "scale_pos_weight": scale_pos_weight,
    }

    model = lgb.train(params, lgb.Dataset(X_train, label=y_train), num_boost_round=200)
    return model, params

def train_catboost(trial, X_train, y_train, class_weights):

    bootstrap_type = trial.suggest_categorical(
        "bootstrap_type",
        ["Bayesian", "Bernoulli"]
    )

    params = {
        "depth": trial.suggest_int("depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.15, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 5, log=True),
        "random_strength": trial.suggest_float("random_strength", 1, 10),
        "iterations": 300,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "verbose": False,
        "class_weights": class_weights,
        "bootstrap_type": bootstrap_type,
    }

    if bootstrap_type == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 5)

    else:
        params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

    model = CatBoostClassifier(**params)
    model.fit(
        X_train,
        y_train,
        verbose=False,
        use_best_model=False,
        early_stopping_rounds=30
    )

    return model, params


##################################################
# ---------------- MASTER FUNCTION ---------------
##################################################

def run_experiment(model_type="lightgbm", number_of_trials=20):

    oversamplers = {
        "none_scale_pos_weight": None,
        "random_oversampler": RandomOverSampler(random_state=42),
        "smote": SMOTE(random_state=42),
        "adasyn": ADASYN(random_state=42)
    }

    root_dir = Path("../data/results")
    root_dir.mkdir(parents=True, exist_ok=True)

    # Load features
    X = np.load("../data/vectors/X_features.npy")
    y = np.load("../data/vectors/y_labels.npy")
    X = impute_nans(X)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root_dir / f"run_{model_type}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------
    # Loop over sampling strategies
    # -------------------------------------------------------
    for sampler_name, sampler in oversamplers.items():
        print(f"\n===== Running sampler: {sampler_name} ({model_type}) =====")

        fold_auc_scores = []
        fold_results = []

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            print(f" Fold {fold+1}/5")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Remove constant cols
            X_train, X_val, mask = remove_constant_cols_local(X_train, X_val)

            # Scale
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

            # Oversampling
            if sampler is not None:
                X_train, y_train = sampler.fit_resample(X_train, y_train)
                class_weights = [1.0, 1.0]
                scale_pos_weight = 1.0
            else:
                pos = sum(y_train == 1)
                neg = sum(y_train == 0)
                scale_pos_weight = neg / pos
                class_weights = [1.0, scale_pos_weight]

            # ----------------------------------------------------
            #  OPTUNA Objective (shared wrapper)
            # ----------------------------------------------------
            def objective(trial):
                if model_type == "lightgbm":
                    model, _ = train_lightgbm(trial, X_train, y_train, scale_pos_weight)
                    preds = model.predict(X_val)

                elif model_type == "catboost":
                    model, _ = train_catboost(trial, X_train, y_train, class_weights)
                    preds = model.predict_proba(X_val)[:, 1]

                return roc_auc_score(y_val, preds)

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=number_of_trials)

            best_params = study.best_params

            #train final model with best params

            if model_type == "lightgbm":
                best_params.update({
                    "objective": "binary",
                    "metric": "auc",
                    "verbosity": -1,
                    "boosting_type": "gbdt",
                    "scale_pos_weight": scale_pos_weight
                })
                final_model = lgb.train(best_params, lgb.Dataset(X_train, label=y_train), num_boost_round=300)
                val_preds = final_model.predict(X_val)

            else:  # CatBoost
                best_params.update({
                    "iterations": 400,
                    "loss_function": "Logloss",
                    "eval_metric": "AUC",
                    "verbose": False,
                    "class_weights": class_weights,
                })
                final_model = CatBoostClassifier(**best_params)
                final_model.fit(X_train, y_train, verbose=False)
                val_preds = final_model.predict_proba(X_val)[:, 1]

            auc = roc_auc_score(y_val, val_preds)
            fpr, tpr, thr = roc_curve(y_val, val_preds)
            youden = tpr - fpr
            best_idx = np.argmax(youden)
            best_threshold = thr[best_idx]
            best_youden = youden[best_idx]

            pred_labels = (val_preds >= best_threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val, pred_labels).ravel()

            precision = precision_score(y_val, pred_labels)
            recall = recall_score(y_val, pred_labels)
            f1 = f1_score(y_val, pred_labels)

            fold_results.append({
                "fold": fold,
                "auc": float(auc),
                "best_threshold": float(best_threshold),
                "best_youden": float(best_youden),
                "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "best_params": best_params,
            })

            fold_auc_scores.append(auc)

            # save fold results
            fold_dir = run_dir / f"{sampler_name}_fold{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            if model_type == "lightgbm":
                final_model.save_model(fold_dir / "model_lightgbm.txt")
            else:
                final_model.save_model(fold_dir / "model_catboost.cbm")

            joblib.dump(scaler, fold_dir / "scaler.pkl")
            np.save(fold_dir / "val_preds.npy", val_preds)
            np.save(fold_dir / "y_val.npy", y_val)

            with open(fold_dir / "metrics.json", "w") as f:
                json.dump(fold_results[-1], f, indent=4)

        # save summary
        summary = {
            "sampler": sampler_name,
            "model_type": model_type,
            "mean_auc": float(np.mean(fold_auc_scores)),
            "std_auc": float(np.std(fold_auc_scores)),
            "folds": fold_results
        }

        with open(run_dir / f"{sampler_name}_summary.json", "w") as f:
            json.dump(summary, f, indent=4)

    logging.info(f"\nTraining complete for all sampling strategies ({model_type})")


if __name__ == "__main__":
    # run_experiment(model_type="lightgbm")
    run_experiment(model_type="catboost")