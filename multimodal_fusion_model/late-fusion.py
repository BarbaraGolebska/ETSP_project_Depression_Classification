import json
import pickle
import joblib
import sys
from pathlib import Path
import numpy as np
import optuna
import pandas as pd
import lightgbm as lgb
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from audio_based_classifier.tree_models.utils import ModelPipeline

OPTUNA_N_TRIALS = 2000
OPTUNA_STORAGE_PATH = "./late-fusion_optuna_journal_storage.log"

# optuna related functions

def get_optuna_storage():
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(OPTUNA_STORAGE_PATH),
    )
    return storage


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

def get_audio_based_datasets(model_name):
    
    set_names = ["X_train", "y_train", "X_dev", "y_dev", "X_test", "y_test"]

    return [np.load(f"../data/audio/{model_name}/{set_name}.npy") for set_name in set_names]


def get_audio_based_models():
    # declaring as lists to be able to add other models later

    model_files = ["../audio_based_classifier/results/lightgbm_smote_hubert_mfcc_egamps.pkl"]
    # get the names for the models, which would be the stemmed filenames
    model_names = [Path(f).stem for f in model_files]
    # get the models themselves
    models = [joblib.load(model_file) for model_file in model_files]
    # get the thresholds for the models
    thresholds = [model.best_threshold for model in models]
    
    # combine all three together
    return list(zip(model_names, models, thresholds))

# functions where we use one modality at a time

def get_predictions_dict(models, X):
    predictions_dict = dict()
    for model_tuple in models:
        model_name, model, threshold = model_tuple
        print(model_name, model, threshold)
        probas = model.predict(X)
        predictions = (probas >= threshold).astype(int)
        predictions_dict[model_name] = predictions
    return predictions_dict

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


def main():
    X_train_text, y_train_text, X_dev_text, y_dev_text, X_test_text, y_test_text = get_text_based_datasets()
    
    #for audio since we use a little bit different data it needs to loaded more than once
    X_train_audio, y_train_audio, X_dev_audio, y_dev_audio, X_test_audio, y_test_audio = get_audio_based_datasets('lightgbm_smote_hubert_mfcc_egamps')

    # text_based_models = get_text_based_models()
    audio_based_models = get_audio_based_models()

    # predictions_dict_text = get_predictions_dict(text_based_models, X_test_text)
    predictions_dict_audio = get_predictions_dict(audio_based_models, X_test_audio)
    # unite two dicts
    predictions_dict = predictions_dict_audio | predictions_dict_audio

    # equal weighting
    model_weights = {
        "embeddings-based_MLP": 1,
        "embeddings-based_CatBoost": 1,
        "embeddings-based_LR": 1,
        "lightgbm_smote_hubert_mfcc_egamps": 1,
    }
    hard_weighted_predictions, soft_weighted_predictions = weighted_vote(predictions_dict, model_weights)

    # get results
    report = classification_report(y_test_text, hard_weighted_predictions)
    print(report)
    # confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(y_test_text, hard_weighted_predictions)
    plt.show()

    # use optuna to optimize weighting

    def objective(trial, y_true):
        model_weights = {
            "embeddings-based_MLP": trial.suggest_float("embeddings-based_MLP", 0, 1),
            "embeddings-based_CatBoost": trial.suggest_float("embeddings-based_CatBoost", 0, 1),
            "embeddings-based_LR": trial.suggest_float("embeddings-based_LR", 0, 1),
            "model_lightgbm": trial.suggest_float("model_lightgbm", 0, 1),
        }
        _, proba = weighted_vote(predictions_dict, model_weights)
        return roc_auc_score(y_true, proba)


    sampler = optuna.samplers.TPESampler(seed=42) # for reproducibility
    study_name = f"late-fusion"
    study = optuna.create_study(
        study_name=study_name,
        storage=get_optuna_storage(),
        direction='maximize',
        sampler=sampler,
        load_if_exists=True
    )
    study.optimize(lambda trial: objective(trial, y_test_text), n_trials=OPTUNA_N_TRIALS)

    best_weights = study.best_params
    print(best_weights)

    hard_weighted_predictions, soft_weighted_predictions = weighted_vote(predictions_dict, best_weights)

    # get results
    report = classification_report(y_test_text, hard_weighted_predictions)
    print(report)
    # confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(y_test_text, hard_weighted_predictions)
    plt.show()


if __name__ == "__main__":
    main()
