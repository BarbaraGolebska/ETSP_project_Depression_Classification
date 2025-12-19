import json
import pickle
import joblib
import sys
from pathlib import Path
from matplotlib.artist import get
import numpy as np
import optuna
import pandas as pd
import lightgbm as lgb
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_auc_score, roc_curve,precision_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import audio_based_classifier.project_utils as project_utils
sys.modules["project_utils"] = project_utils

from audio_based_classifier.project_utils import BaselineLinearWrapper
from audio_based_classifier.tree_models.utils import ModelPipeline


OPTUNA_N_TRIALS = 2000
OPTUNA_STORAGE_PATH = "./late-fusion_optuna_journal_storage.log"

# optuna related functions

def get_optuna_storage():
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(OPTUNA_STORAGE_PATH),
    )
    return storage

def get_optuna_storage_rdb():
    return optuna.storages.RDBStorage(
        url="sqlite:///late_fusion_optuna.db",
        engine_kwargs={"connect_args": {"check_same_thread": False}}
    )

# text based functions

def split(df):
    return (
        df[df.split == "train"].reset_index(drop=True),
        df[df.split == "dev"].reset_index(drop=True),
        df[df.split == "test"].reset_index(drop=True),
    )


def extract(df):
    return np.vstack(df["embedding"].to_numpy()), df["target_depr"].astype(int).to_numpy()

def get_text_based_datasets():
    embeddings_df = pd.read_csv("../data/processed/embeddings.csv", index_col=0,
                                converters={"embedding": lambda s:
                                np.fromstring(s.strip("[]"), sep=" ")})  # make sure embeddings are numpy array
    train_df, dev_df, test_df = split(embeddings_df)
    X_train, y_train = extract(train_df)
    X_dev, y_dev = extract(dev_df)
    X_test, y_test = extract(test_df)
    return X_train, y_train, X_dev, y_dev, X_test, y_test


def get_text_based_models():
    path_to_models = Path("../text_based_classifier/results")
    # grab all embeddings based models
    model_files = [str(p) for p in path_to_models.glob("embeddings-based_*.pkl")]
    # get the names for the models, which would be the stemmed filenames
    model_names = [Path(f).stem for f in model_files]
    # get the models themselves
    models = [joblib.load(model_file) for model_file in model_files]
    # get the thresholds for the models
    with open("../text_based_classifier/results/embeddings_thresholds.pkl", 'rb') as f:
        thresholds_dict = pickle.load(f)
    # assign thresholds in the same order as model_names
    thresholds = [thresholds_dict[model_name] for model_name in model_names]
    # combine all three together
    return list(zip(model_names, models, thresholds))


# audio based functions

def load_processed_audio_data(df: pd.DataFrame) -> list[pd.DataFrame]:

    df_train = df[df["split"] == "train"]
    df_dev = df[df["split"] == "dev"]
    df_test = df[df["split"] == "test"]

    drop_cols = ["participant_id", "target_depr", "target_ptsd", "split"]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train.drop(columns=drop_cols))
    y_train = df_train["target_depr"].values

    X_dev = scaler.transform(df_dev.drop(columns=drop_cols))
    y_dev = df_dev["target_depr"].values
    
    X_test = scaler.transform(df_test.drop(columns=drop_cols))
    y_test = df_test["target_depr"].values
    
    return X_train, y_train, X_dev, y_dev, X_test, y_test

def get_audio_based_datasets_lightgbm(model_name):
    
    set_names = ["X_train", "y_train", "X_dev", "y_dev", "X_test", "y_test"]

    return [np.load(f"../data/processed/audio/{model_name}/{set_name}.npy") for set_name in set_names]

def get_audio_based_datasets():
    embeddings_df = pd.read_csv("../data/processed/audio/hubert/hubert_aggregated_embeddings.csv") 
    X_train, y_train, X_dev, y_dev, X_test, y_test = load_processed_audio_data(embeddings_df)
    return X_train, y_train, X_dev, y_dev, X_test, y_test

def get_audio_based_models():
    
    # declaring as lists to be able to add other models later

    model_files = ["../audio_based_classifier/results/lightgbm_smote_hubert_mfcc_egemaps.pkl","../audio_based_classifier/results/hubert_None_baseline.pkl"]
    # get the names for the models, which would be the stemmed filenames
    model_names = [Path(f).stem for f in model_files]
    # get the models themselves
    # models = [joblib.load(model_file) for model_file in model_files]
    models = []
    for model_file in model_files:
        print("Loading:", model_file)
        models.append(joblib.load(model_file))
    # get the thresholds for the models
    thresholds = [model.best_threshold for model in models]
    
    # combine all three together
    return list(zip(model_names, models, thresholds))

# functions where we use one modality at a time

def get_predictions_dict(models, X):
    predictions_dict = dict()
    probas_dict = dict()
    for model_tuple in models:
        model_name, model, threshold = model_tuple
        print(model_name, model, threshold)
        probas = model.predict(X)
        predictions = (probas >= threshold).astype(int)
        predictions_dict[model_name] = predictions
        probas_dict[model_name] = probas
    return predictions_dict, probas_dict

# voting functions

def weighted_vote(predictions, model_weights, final_threshold=0.5):
    model_names = list(model_weights.keys())
    total_weight = sum(model_weights.values())
    predictions_length = len(predictions[model_names[0]])

    soft_weighted_predictions = []
    hard_weighted_predictions = []

    for i in range (0, predictions_length):
        prediction_weighted_sum = 0
        for model_name in model_names:
            prediction_weighted_sum += predictions[model_name][i] * model_weights[model_name]
        prediction_weighted_mean = prediction_weighted_sum / total_weight
        soft_weighted_predictions.append(prediction_weighted_mean)
        final_prediction = int(prediction_weighted_mean >= final_threshold)
        hard_weighted_predictions.append(final_prediction)

    return hard_weighted_predictions, soft_weighted_predictions


def log_callback(study, trial):
    with open("late_fusion_optuna_12_10.log", "a", encoding="utf-8") as f:
        f.write(
            f"Trial {trial.number} | Value: {trial.value} | Params: {trial.params}\n"
        )

def build_meta_X_from_dicts(*probas_dicts, model_order=None, sort_keys=True):

    merged = {}
    for d in probas_dicts:
        if d is None:
            continue
        merged.update(d)

    if model_order is None:
        model_order = sorted(merged.keys()) if sort_keys else list(merged.keys())
    else:
        missing = [k for k in model_order if k not in merged]
        if missing:
            raise KeyError(f"Missing probas for models: {missing}")

    X_meta = np.column_stack([np.asarray(merged[k]).reshape(-1) for k in model_order])
    return X_meta, model_order

def add_interaction_terms(probas_dict, interactions):

    new_dict = probas_dict.copy()

    for m1, m2 in interactions:
        if m1 not in probas_dict or m2 not in probas_dict:
            raise KeyError(f"Missing models for interaction: {m1}, {m2}")

        interaction_name = f"{m1}x{m2}"
        new_dict[interaction_name] = probas_dict[m1] * probas_dict[m2]

    return new_dict


def thresholds_youden_and_fixed(y_true, p):

    y_true = np.asarray(y_true).astype(int).reshape(-1)
    p = np.asarray(p).reshape(-1)

    fpr, tpr, thr = roc_curve(y_true, p)

    #roc_curve sometimes returns thr[0] = inf; drop non-finite thresholds
    mask = np.isfinite(thr)
    fpr, tpr, thr = fpr[mask], tpr[mask], thr[mask]

    j = tpr - fpr
    best_idx = int(np.argmax(j))

    youden_thr = float(thr[best_idx])
    youden_j = float(j[best_idx])

    print("best youden ", youden_j)
    return youden_thr, youden_j


def main():
    X_train_text, y_train_text, X_dev_text, y_dev_text, X_test_text, y_test_text = get_text_based_datasets()
    
    #for audio since we use a little bit different data it needs to loaded more than once
    X_train_audio_l, y_train_audio_l, X_dev_audio_l, y_dev_audio_l, X_test_audio_l, y_test_audio_l = get_audio_based_datasets_lightgbm('lightgbm_smote_hubert_mfcc_egemaps')
    X_train_audio, y_train_audio, X_dev_audio, y_dev_audio, X_test_audio, y_test_audio = get_audio_based_datasets()
    
    text_based_models = get_text_based_models()
    audio_based_models_lightgbm = get_audio_based_models()[:1]
    audio_based_models_logreg = get_audio_based_models()[1:]

    predictions_dict_text_dev, probas_dict_text_dev = get_predictions_dict(text_based_models, X_dev_text)
    predictions_dict_audio_lightgbm_dev, probas_dict_audio_lightgbm_dev = get_predictions_dict(audio_based_models_lightgbm, X_dev_audio_l)
    predictions_dict_audio_logreg_dev, probas_dict_audio_logreg_dev = get_predictions_dict(audio_based_models_logreg, X_dev_audio)
    # unite the 3 di
    predictions_dict_dev = predictions_dict_text_dev | predictions_dict_audio_lightgbm_dev | predictions_dict_audio_logreg_dev
    
    predictions_dict_text_test, probas_dict_text_test = get_predictions_dict(text_based_models, X_test_text)
    predictions_dict_audio_lightgbm_test, probas_dict_audio_lightgbm_test = get_predictions_dict(audio_based_models_lightgbm, X_test_audio_l)
    predictions_dict_audio_logreg_test, probas_dict_audio_logreg_test = get_predictions_dict(audio_based_models_logreg, X_test_audio)
    # unite two dicts
    predictions_dict_test = predictions_dict_text_test | predictions_dict_audio_lightgbm_test | predictions_dict_audio_logreg_test
    

    # equal weighting
    model_weights = {
        "embeddings-based_MLP": 1,
        "embeddings-based_CatBoost": 1,
        "embeddings-based_LR": 1,
        "lightgbm_smote_hubert_mfcc_egemaps": 1,
        "hubert_None_baseline": 1,
    }
    
    hard_weighted_predictions, soft_weighted_predictions = weighted_vote(predictions_dict_dev, model_weights)

    # get results
    report = classification_report(y_dev_text, hard_weighted_predictions)
    print(report)
    # confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(y_dev_text, hard_weighted_predictions)
    plt.show()
    

    # use optuna to optimize weighting

    def objective(trial, y_true):
        model_weights = {
            "embeddings-based_MLP": trial.suggest_float("embeddings-based_MLP", 0, 1),
            "embeddings-based_CatBoost": trial.suggest_float("embeddings-based_CatBoost", 0, 1),
            "embeddings-based_LR": trial.suggest_float("embeddings-based_LR", 0, 1),
            "lightgbm_smote_hubert_mfcc_egemaps": trial.suggest_float("lightgbm_smote_hubert_mfcc_egemaps", 0, 1),
            "hubert_None_baseline": trial.suggest_float("hubert_None_baseline", 0, 1),
        }
        _, proba = weighted_vote(predictions_dict_dev, model_weights)
        return roc_auc_score(y_true, proba)


    sampler = optuna.samplers.TPESampler(seed=42) # for reproducibility
    study_name = f"late-fusion"
    study = optuna.create_study(
        study_name=study_name,
        storage=get_optuna_storage_rdb(),
        direction='maximize',
        sampler=sampler,
        load_if_exists=True
    )
    
    study.optimize(
            lambda trial: objective(trial, y_dev_text),
            n_trials=OPTUNA_N_TRIALS,
            callbacks=[log_callback]
        )

    best_weights = study.best_params
    print(best_weights)
    
    hard_weighted_predictions, soft_weighted_predictions = weighted_vote(predictions_dict_test, model_weights)
    print("\n[TEST] Late fusion with equal weights")
    report = classification_report(y_test_text, hard_weighted_predictions)
    print(report)
    # confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(y_test_text, hard_weighted_predictions)
    plt.show()

    hard_weighted_predictions, soft_weighted_predictions = weighted_vote(predictions_dict_test, best_weights)

    # get results
    print("\n[TEST] Late fusion with optimized weights")
    report = classification_report(y_test_text, hard_weighted_predictions)
    print(report)
    # confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(y_test_text, hard_weighted_predictions)
    plt.show()
    
    
    probas_dev_all = (
    probas_dict_text_dev
    | probas_dict_audio_lightgbm_dev
    | probas_dict_audio_logreg_dev
)

#     interaction_pairs = [
#     ("hubert_None_baseline", "lightgbm_smote_hubert_mfcc_egemaps"),
# ]

#     probas_dev_all = add_interaction_terms(probas_dev_all, interaction_pairs)

    X_meta_dev, model_order = build_meta_X_from_dicts(
        probas_dev_all,
        model_order=None
    )

    y_meta_dev = y_dev_text.astype(int)

    meta_clf = LogisticRegression(
        solver="liblinear",
        penalty="l2", 
        class_weight="balanced",   
        max_iter=10000
    )

    meta_clf.fit(X_meta_dev, y_meta_dev)

    p_meta_dev = meta_clf.predict_proba(X_meta_dev)[:, 1]
    youden_thr, youden_j = thresholds_youden_and_fixed(y_meta_dev, p_meta_dev)
    print("DEV Youden thr:", youden_thr, "J:", youden_j)
    
    # ---------- EVALUATION ON TEST ----------
    probas_test_all = (
    probas_dict_text_test
    | probas_dict_audio_lightgbm_test
    | probas_dict_audio_logreg_test
)

    # probas_test_all = add_interaction_terms(probas_test_all, interaction_pairs)

    X_meta_test, _ = build_meta_X_from_dicts(
        probas_test_all,
        model_order=model_order
    )

    y_meta_test = y_test_text.astype(int)
    p_meta_test = meta_clf.predict_proba(X_meta_test)[:, 1]

    yhat_test_dev_youden = (p_meta_test >= youden_thr).astype(int)
    print("\n[TEST] report at DEV Youden threshold")
    print(classification_report(y_meta_test, yhat_test_dev_youden, digits=4))
    ConfusionMatrixDisplay.from_predictions(y_meta_test, yhat_test_dev_youden)
    plt.title("TEST at DEV Youden threshold")
    plt.show()   

    yhat_test_06 = (p_meta_test >= 0.6).astype(int)
    print("\n[TEST] report at threshold 0.6")
    print(classification_report(y_meta_test, yhat_test_06, digits=4))
    ConfusionMatrixDisplay.from_predictions(y_meta_test, yhat_test_06)
    plt.title("TEST at threshold 0.6")
    plt.show()
    
    
if __name__ == "__main__":
    main()

