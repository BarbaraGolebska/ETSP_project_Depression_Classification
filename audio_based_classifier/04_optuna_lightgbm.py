import pandas as pd
import numpy as np
import random
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE
import lightgbm as lgb

# =========================
# REPRODUCIBILITY
# =========================
def set_seed(seed=1):
    np.random.seed(seed)
    random.seed(seed)

set_seed(1)

# =========================
# METRICS
# =========================
def compute_best_youden(y_true, y_proba):
    thresholds = np.unique(y_proba)
    best_youden = -1
    best_threshold = 0

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn + 1e-12)
        specificity = tn / (tn + fp + 1e-12)
        youden = sensitivity + specificity - 1

        if youden > best_youden:
            best_youden = youden
            best_threshold = thr

    return best_youden, best_threshold

# =========================
# OPTUNA OBJECTIVE
# =========================
def objective(trial, X, y, kf, oversampler_name):
    
    params = { "max_depth": 3, # shallower tree 
              "num_leaves": 4, # <= 2^max_depth
            "min_child_samples": 1, # smaller minimum child samples 
            "learning_rate": 0.05, 
            "subsample": 0.8, 
            "colsample_bytree": 0.8, 
            "reg_alpha": 0.0, 
            "reg_lambda": 1.0, 
            "n_estimators": 100, 
            "random_state": 42, 
            "device": "cpu", 
            "verbosity": -1
              }
    
    fold_scores = []

    for train_idx, val_idx in kf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Oversampling
        if oversampler_name == "RandomOverSampler":
            sampler = RandomOverSampler(random_state=42)
        elif oversampler_name == "SMOTE":
            sampler = SMOTE(random_state=42)
        elif oversampler_name == "BorderlineSMOTE":
            sampler = BorderlineSMOTE(random_state=42)
        else:
            sampler = None

        if sampler is not None:
            X_train, y_train = sampler.fit_resample(X_train, y_train)

        # Train LightGBM
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_val)[:, 1]
        ydn, _ = compute_best_youden(y_val, y_proba)
        fold_scores.append(ydn)

    return np.mean(fold_scores)

# =========================
# MAIN FUNCTION
# =========================
ftypes = {
    "expert_k": "ExpertK_aggregated_features.csv",
    #"all": "merged_all_features.csv"
}

#oversampling_methods = ["None", "RandomOverSampler", "SMOTE", "BorderlineSMOTE"]
oversampling_methods = ["RandomOverSampler"]


def main():
    scaler = StandardScaler()

    for ftype, path in ftypes.items():
        df = pd.read_csv(f"data/processed/{path}")
        df_train, df_dev = df[df["split"] == "train"], df[df["split"] == "dev"]

        drop_cols = ["participant_id", "target_depr", "target_ptsd", "split"]
        X_train = scaler.fit_transform(df_train.drop(columns=drop_cols))
        y_train = df_train["target_depr"].values

        X_dev = scaler.transform(df_dev.drop(columns=drop_cols))
        y_dev = df_dev["target_depr"].values
        rng = np.random.RandomState(42)

        for oversampler_name in oversampling_methods:
            print(f"\n========== {ftype.upper()} | Oversampling: {oversampler_name} ==========")

            # OPTUNA STUDY
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
            set_seed(1)
            sampler = optuna.samplers.TPESampler(seed=rng.randint(0, 10**6))
            #sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(lambda trial: objective(trial, X_train, y_train, kf, oversampler_name), n_trials=20)

            best_params = study.best_params
            print(f"Best params: {best_params}, Best Youden: {study.best_value:.4f}")

            # Train final model
            if oversampler_name != "None":
                if oversampler_name == "RandomOverSampler":
                    sampler = RandomOverSampler(random_state=42)
                elif oversampler_name == "SMOTE":
                    sampler = SMOTE(random_state=42)
                else:
                    sampler = BorderlineSMOTE(random_state=42)
                X_train_os, y_train_os = sampler.fit_resample(X_train, y_train)
            else:
                X_train_os, y_train_os = X_train, y_train

            final_model = lgb.LGBMClassifier(**best_params, random_state=42, n_jobs=-1, device="gpu")
            final_model.fit(X_train_os, y_train_os)

            # Feature importances
            importances = final_model.feature_importances_
            feature_names = df_train.drop(columns=drop_cols).columns
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)

            print("\nTop 15 most important features:")
            print(importance_df.head(15))

            # Evaluate on dev set
            y_proba = final_model.predict_proba(X_dev)[:, 1]
            ydn_dev, best_threshold = compute_best_youden(y_dev, y_proba)
            y_pred = (y_proba >= best_threshold).astype(int)
            cm = confusion_matrix(y_dev, y_pred)

            print(f"\nFinal model Youden on dev set: {ydn_dev:.4f} at threshold={best_threshold:.3f}")
            print(f"Confusion Matrix:\n{cm}")

            fn_mask = (y_dev == 1) & (y_pred == 0)
            misdiagnosed = df_dev.loc[fn_mask, "participant_id"].values
            print("Participant IDs misdiagnosed (False Negatives):", misdiagnosed)

            import pickle

            # Save 
            model_path= f"audio_based_classifier/models/{ftype}_{oversampler_name}_lgbm.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(final_model, f)
            print(f"Model saved to {model_path}")

            # Load
            #with open('lightgbm_model.pkl', 'rb') as f:
            #    loaded_model = pickle.load(f)


if __name__ == "__main__":
    main()
