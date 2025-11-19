from pathlib import Path
import numpy as np
import optuna
from imblearn.under_sampling import RandomUnderSampler
from joblib import Parallel, delayed
from nltk import sent_tokenize
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
from deepmultilingualpunctuation import PunctuationModel
from sentence_transformers import SentenceTransformer
from catboost import CatBoostClassifier
from imblearn.pipeline import make_pipeline

RESULTS_DIR = "./results"

OPTUNA_N_TRIALS = 100
OPTUNA_STORAGE_PATH = "./embeddings-based_optuna_journal_storage.log"

PUNCTUATION_MODEL = PunctuationModel()
MPNET_MODEL = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


# dataset related functions

def load_dataset(path="../data/processed/text_combined.csv"):
    return pd.read_csv(path, index_col=0)


def split(df):
    return (
        df[df.split == "train"].reset_index(drop=True),
        df[df.split == "dev"].reset_index(drop=True),
        df[df.split == "test"].reset_index(drop=True),
    )


def extract(df):
    return df["text"], df["target_depr"]


# model related functions

def get_pipeline(model_params, model_name):
    punctuation_transformer = FunctionTransformer(lambda docs: docs.apply(PUNCTUATION_MODEL.restore_punctuation))
    sentence_splitter_transformer = FunctionTransformer(lambda docs: docs.apply(sent_tokenize))
    sentence_embedding_transformer = FunctionTransformer(lambda docs: docs.apply(MPNET_MODEL.encode))
    mean_pooling_transformer = FunctionTransformer(lambda docs: docs.apply(lambda x: np.mean(x, axis=0)))
    to_matrix_transformer = FunctionTransformer(np.vstack)
    undersampler = RandomUnderSampler(random_state=42)

    if model_name=="CatBoost":
        model = CatBoostClassifier(verbose=0, random_state=42, **model_params)
    elif model_name=="MLP":
        model = MLPClassifier(random_state=42, max_iter=500, **model_params)

    return make_pipeline(punctuation_transformer, sentence_splitter_transformer, sentence_embedding_transformer,
                         mean_pooling_transformer, to_matrix_transformer, undersampler, model)


def train_evaluate(X_train, y_depr_train, model_params, model_name):
    scores = []

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for train_index, test_index in cv.split(X_train, y_depr_train):
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_depr_train.iloc[train_index], y_depr_train.iloc[test_index]

        pipeline = get_pipeline(model_params, model_name)
        pipeline.fit(X_train_fold, y_train_fold)

        predictions = pipeline.predict_proba(X_test_fold)[:, 1]
        scores.append(roc_auc_score(y_test_fold, predictions))

    return np.mean(scores)


# optuna related functions

def get_optuna_storage():
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(OPTUNA_STORAGE_PATH),
    )
    return storage


def objective(trial, X_train, y_depr_train, model_name):
    if model_name == "CatBoost":
        model_params = {
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
            "depth": trial.suggest_int("depth", 1, 12),
            "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            ),
        }
        if model_params["bootstrap_type"] == "Bayesian":
            model_params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
        elif model_params["bootstrap_type"] == "Bernoulli":
            model_params["subsample"] = trial.suggest_float("subsample", 0.1, 1)
    elif model_name == "MLP":
        n_layers = trial.suggest_int('n_layers', 1, 4)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f"n_units_l{i}", 16, 256, step=16))
        model_params = {
            "hidden_layer_sizes": tuple(layers),
            "activation": trial.suggest_categorical(
                "activation", ["relu", "tanh", "logistic"]
            ),
            "solver": trial.suggest_categorical(
                "solver", ["adam", "lbfgs"]
            ),
            "alpha": trial.suggest_float(
                "alpha", 1e-5, 1e-1, log=True
            ),
            "learning_rate_init": trial.suggest_float(
                "learning_rate_init", 1e-4, 1e-1, log=True
            )
        }

    # store the cleaned params on the trial
    trial.set_user_attr("model_params", model_params)

    return train_evaluate(X_train, y_depr_train, model_params, model_name)


def run_optimization(X_train, y_depr_train, optuna_study_name, model_name):
    study = optuna.load_study(study_name=optuna_study_name, storage=get_optuna_storage())
    study.optimize(lambda trial: objective(trial, X_train, y_depr_train, model_name), n_trials=1)


def main():
    df = load_dataset()
    train_df, dev_df, test_df = split(df)
    X_train, y_train = extract(train_df)
    X_dev, y_dev = extract(dev_df)

    # evaluate two models

    model_names = ["MLP", "CatBoost"]
    for model_name in model_names:

        # run hyperparameter optimization
        study_name = f"embeddings-based_{model_name}"

        optuna.create_study(study_name=study_name, storage=get_optuna_storage(), direction='maximize', load_if_exists=True)

        Parallel(n_jobs=1, backend='multiprocessing')(
            delayed(run_optimization)(X_train, y_train, study_name, model_name)
            for _ in range(OPTUNA_N_TRIALS)
        )

        # get the results from hyperparameter optimization
        study = optuna.load_study(study_name=study_name, storage=get_optuna_storage())

        # get the pipeline wth the chosen parameters
        best_params = study.best_trial.user_attrs["model_params"]
        pipeline = get_pipeline(best_params, model_name)

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_dev)

        # save results
        results_dir = Path(RESULTS_DIR)
        results_dir.mkdir(parents=True, exist_ok=True)

        with open(results_dir / f"{study_name}_results.txt", "w") as f:
            f.write(f"Embeddings + {model_name}:\n")
            f.write(f"Best hyperparameters: {study.best_trial.params}\n")
            f.write("\nClassification Report:\n")
            f.write(classification_report(y_dev, y_pred))
            f.write("\n\n")

        # save confusion matrices
        disp = ConfusionMatrixDisplay.from_predictions(y_dev, y_pred)
        disp.figure_.savefig(results_dir / f"embeddings-based_{model_name}_cm.png")


if __name__ == "__main__":
    main()
