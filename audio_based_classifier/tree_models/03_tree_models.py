import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime
import logging

import optuna
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
        "random_oversampler": RandomOverSampler(random_state=42),
        "smote": SMOTE(random_state=42),
        "adasyn": ADASYN(random_state=42)
    }

    root_dir = Path("../data/results")
    root_dir.mkdir(parents=True, exist_ok=True)

    out_dir = Path("../data/vectors/new")

    X_train = np.load(out_dir / "X_train.npy")
    y_train = np.load(out_dir / "y_train.npy")
    X_dev = np.load(out_dir / "X_dev.npy")
    y_dev = np.load(out_dir / "y_dev.npy")
    X_test = np.load(out_dir / "X_test.npy")
    y_test = np.load(out_dir / "y_test.npy")

    X_train = impute_nans(X_train)
    X_dev = impute_nans(X_dev)
    X_test = impute_nans(X_test)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root_dir / f"run_{model_type}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for sampler_name, sampler in oversamplers.items():
        logging.info(f"\n===== Running sampler: {sampler_name} ({model_type}) =====")

        X_tr = X_train.copy()
        y_tr = y_train.copy()
        X_val = X_dev.copy()
        y_val = y_dev.copy()

        X_tr, X_val, mask = remove_constant_cols_local(X_tr, X_val)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)

        if sampler is not None:
            X_tr, y_tr = sampler.fit_resample(X_tr, y_tr)
            class_weights = [1.0, 1.0]
            scale_pos_weight = 1.0
        else:
            pos = sum(y_tr == 1)
            neg = sum(y_tr == 0)
            scale_pos_weight = neg / pos
            class_weights = [1.0, scale_pos_weight]

        # ---------------- OPTUNA ----------------
        def objective(trial):
            if model_type == "lightgbm":
                model, _ = train_lightgbm(trial, X_tr, y_tr, scale_pos_weight)
                preds = model.predict(X_val)
            else:
                model, _ = train_catboost(trial, X_tr, y_tr, class_weights)
                preds = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, preds)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=number_of_trials)

        best_params = study.best_params

        # ---------------- FINAL MODEL ----------------
        if model_type == "lightgbm":
            best_params.update({
                "objective": "binary",
                "metric": "auc",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "scale_pos_weight": scale_pos_weight
            })
            final_model = lgb.train(best_params, lgb.Dataset(X_tr, label=y_tr), num_boost_round=300)
            dev_preds = final_model.predict(X_val)
        else:
            best_params.update({
                "iterations": 400,
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "verbose": False,
                "class_weights": class_weights,
            })
            final_model = CatBoostClassifier(**best_params)
            final_model.fit(X_tr, y_tr, verbose=False)
            dev_preds = final_model.predict_proba(X_val)[:, 1]

        auc = roc_auc_score(y_val, dev_preds)
        fpr, tpr, thr = roc_curve(y_val, dev_preds)
        youden = tpr - fpr
        best_idx = np.argmax(youden)
        best_threshold = thr[best_idx]
        dev_youden = float(youden[best_idx])

        dev_labels = (dev_preds >= best_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, dev_labels).ravel()

        precision = precision_score(y_val, dev_labels)
        recall = recall_score(y_val, dev_labels)
        f1 = f1_score(y_val, dev_labels)

        logging.info("----- TEST EVALUATION -----")
        X_te = X_test[:, mask]
        X_te = scaler.transform(X_te)

        if model_type == "lightgbm":
            test_preds = final_model.predict(X_te)
        else:
            test_preds = final_model.predict_proba(X_te)[:, 1]

        test_labels = (test_preds >= best_threshold).astype(int)

        auc_test = roc_auc_score(y_test, test_preds)
        tn2, fp2, fn2, tp2 = confusion_matrix(y_test, test_labels).ravel()

        fpr2, tpr2, _ = roc_curve(y_test, test_preds)
        test_youden = float(max(tpr2 - fpr2))

        precision_test = precision_score(y_test, test_labels)
        recall_test = recall_score(y_test, test_labels)
        f1_test = f1_score(y_test, test_labels)

        sampler_dir = run_dir / sampler_name
        sampler_dir.mkdir(parents=True, exist_ok=True)

        if model_type == "lightgbm":
            final_model.save_model(sampler_dir / "model_lightgbm.txt")
        else:
            final_model.save_model(sampler_dir / "model_catboost.cbm")

        joblib.dump(scaler, sampler_dir / "scaler.pkl")
        np.save(sampler_dir / "dev_preds.npy", dev_preds)
        np.save(sampler_dir / "test_preds.npy", test_preds)

        metrics = {
            "sampler": sampler_name,
            "best_threshold": float(best_threshold),

            "test_auc": float(auc_test),
            "test_youden": test_youden,
            "test_confusion": [int(tn2), int(fp2), int(fn2), int(tp2)],
            "test_precision": float(precision_test),
            "test_recall": float(recall_test),
            "test_f1": float(f1_test),

            "best_params": best_params
        }

        with open(sampler_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

    logging.info("Training complete")


if __name__ == "__main__":
    run_experiment(model_type="lightgbm", number_of_trials=1000)
