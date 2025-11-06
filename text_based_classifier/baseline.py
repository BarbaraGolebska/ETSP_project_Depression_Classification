import numpy as np
import pandas as pd
import re
import optuna
from imblearn.under_sampling import RandomUnderSampler
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, fbeta_score, roc_auc_score, \
    average_precision_score, roc_curve
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

OPTUNA_STUDY_NAME = "baseline_lr"
OPTUNA_N_TRIALS = 20
OPTUNA_STORAGE_PATH = "./baseline_optuna_journal_storage.log"

# NLTK related functions

from pathlib import Path
import nltk

NLTK_DIR = Path("../data/nltk_data")
NLTK_DIR.mkdir(parents=True, exist_ok=True)
nltk.data.path.append(str(NLTK_DIR))

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def nltk_preprocess(text):
    # make sure we get useful tokens
    token_pattern = re.compile(r"(?u)\b[^\W\d_]{3,}\b")  # at least 3 letters, letters only, unicode-aware
    txt = "" if not isinstance(text, str) else text.lower()
    words = token_pattern.findall(txt)
    pos_tags = pos_tag(words)
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    return lemmatized_words


# dataset related functions

def load_dataset(path="../data/processed/text_combined.csv"):
    return pd.read_csv(path, index_col=0)


def get_split(df, split_name):
    """return only rows from the specified split ('train', 'dev', 'test')"""
    return df[df["split"] == split_name].reset_index(drop=True)


def get_X_y_split(df_train, df_dev):
    # get the necessary columns out of df_train
    X_train = df_train["text"]
    y_depr_train = df_train["target_depr"]

    # get the necessary columns out of df_dev
    X_dev = df_dev["text"]
    y_depr_dev = df_dev["target_depr"]

    return X_train, y_depr_train, X_dev, y_depr_dev


# model related functions

def get_pipeline(model_params):
    vectorizer = TfidfVectorizer(tokenizer=nltk_preprocess,
                    ngram_range=(1, 2),
                    min_df=2,  # ignore words that appear in less than 2 documents
                    token_pattern=None,
                )
    undersampler = RandomUnderSampler(random_state=42)
    lr = LogisticRegression(**model_params, random_state=42)

    return make_pipeline(vectorizer, undersampler, lr)


def train_evaluate(X_train, y_depr_train, model_params):
    scores = []

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for train_index, test_index in cv.split(X_train, y_depr_train):
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_depr_train.iloc[train_index], y_depr_train.iloc[test_index]

        pipeline = get_pipeline(model_params)
        pipeline.fit(X_train_fold, y_train_fold)

        predictions = pipeline.predict_proba(X_test_fold)[:, 1]
        scores.append(roc_auc_score(y_test_fold, predictions))

    return np.mean(scores)


def get_thresholded_predictions(probs, threshold):
    return (probs >= threshold).astype(int)


def get_J_threshold(y_true, y_probs):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    j = tpr - fpr
    idx = np.argmax(j[1:]) + 1 # thresholds[0] corresponds to (0,0) point -> inf
    best_threshold = thresholds[idx]
    best_j = j[idx]

    return best_threshold, best_j

def evaluate_with_J_threshold(pipeline, X_dev, y_depr_dev):
    y_probs = pipeline.predict_proba(X_dev)[:, 1]
    best_threshold, best_j = get_J_threshold(y_depr_dev, y_probs)
    y_depr_pred = get_thresholded_predictions(y_probs, best_threshold)

    return y_depr_pred, best_threshold, best_j


# optuna related functions

def get_optuna_storage():
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(OPTUNA_STORAGE_PATH),
    )
    return storage


def objective(trial, X_train, y_depr_train):
    model_params = {'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'newton-cg']),  # small dataset friendly
                    'C': trial.suggest_float("C", 1e-7, 10.0, log=True)}

    return train_evaluate(X_train, y_depr_train, model_params)


def run_optimization(X_train, y_depr_train):
    study = optuna.load_study(study_name="baseline_lr", storage=get_optuna_storage())
    study.optimize(lambda trial: objective(trial, X_train, y_depr_train), n_trials=1)


def main():
    df = load_dataset()

    # get only train part of the dataset
    df_train = get_split(df, "train")
    # get only dev part of the dataset
    df_dev = get_split(df, "dev")

    X_train, y_depr_train, X_dev, y_depr_dev = get_X_y_split(df_train, df_dev)

    # run hyperparameter optimization
    optuna.create_study(study_name=OPTUNA_STUDY_NAME, storage=get_optuna_storage(), direction='maximize', load_if_exists=True)

    Parallel(n_jobs=-1, backend='multiprocessing')(
        delayed(run_optimization)(X_train, y_depr_train)
        for _ in range(OPTUNA_N_TRIALS)
    )

    # get the results from hyperparameter optimization
    study = optuna.load_study(study_name=OPTUNA_STUDY_NAME, storage=get_optuna_storage())

    # get the pipeline wth the chosen parameters
    pipeline = get_pipeline(study.best_trial.params)

    pipeline.fit(X_train, y_depr_train)

    # evaluate on dev set
    y_depr_pred, best_t, best_j = evaluate_with_J_threshold(pipeline, X_dev, y_depr_dev)

    # save results
    results_dir = Path("./results")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "baseline_lr_results.txt", "w") as f:
        f.write(f"Best hyperparameters: {study.best_trial.params}\n")
        f.write(f"Best J statistic on dev set: {best_j} at threshold {best_t}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_depr_dev, y_depr_pred))
        ConfusionMatrixDisplay.from_predictions(y_depr_dev, y_depr_pred)
        plt.savefig(results_dir / "baseline_lr_confusion_matrix.png")
        print(f"Results saved to {results_dir}")

if __name__ == "__main__":
    main()