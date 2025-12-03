import numpy as np
import pandas as pd
import optuna
from optuna.trial import FixedTrial
from imblearn.under_sampling import RandomUnderSampler
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_auc_score, roc_curve
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold

import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin

from tokenizer import nltk_preprocess

OPTUNA_N_TRIALS = 50
OPTUNA_STORAGE_PATH = "./baseline_optuna_journal_storage.log"
# Add a deterministic pruner (median over previous trials)
OPTUNA_PRUNER = optuna.pruners.MedianPruner(
    n_warmup_steps=1,   # don't prune before at least 1 reported step
    n_min_trials=8      # wait for at least 8 completed trials
)

RESULTS_DIR = "./results"

import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


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

# ---- NB-weight transformer (log-count ratio) ----
class NBWeight(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)
        self.r_ = None

    def fit(self, X, y):
        X = sp.csr_matrix(X)
        y = np.asarray(y)
        pos = (y == 1)
        neg = (y == 0)
        if pos.sum() == 0 or neg.sum() == 0:
            raise ValueError("Both classes must be present to compute NB ratios.")
        n_pos = X[pos].sum(axis=0) + self.alpha   # 1 x V
        n_neg = X[neg].sum(axis=0) + self.alpha   # 1 x V
        n_pos = n_pos / n_pos.sum()
        n_neg = n_neg / n_neg.sum()
        r = np.log(n_pos / n_neg)
        self.r_ = np.asarray(r).ravel()
        return self

    def transform(self, X):
        if self.r_ is None:
            raise RuntimeError("NBWeight is not fitted yet.")
        X = sp.csr_matrix(X)
        return X.multiply(self.r_)


# pipelines

def tfidf_pipeline(vectorizer_params, model_params):
    vectorizer = TfidfVectorizer(tokenizer=nltk_preprocess,
                    **vectorizer_params,
                    token_pattern=None,
                )
    undersampler = RandomUnderSampler(random_state=SEED)
    lr = LogisticRegression(**model_params, max_iter=1000, random_state=SEED)

    return ImbPipeline([('vectorizer', vectorizer),
                        ('undersampler', undersampler),
                        ('classifier', lr)])


def nb_pipeline(vectorizer_params, model_params, nb_params):
    vectorizer = CountVectorizer(tokenizer=nltk_preprocess,
                    **vectorizer_params,
                    token_pattern=None,
                )
    nbweight = NBWeight(**nb_params)
    lr = LogisticRegression(**model_params, max_iter=1000, random_state=SEED)

    return ImbPipeline([('vectorizer', vectorizer),
                        ('nbweight', nbweight),
                        ('classifier', lr)])


# evaluation functions

def thresholded_predictions(pipeline, X, threshold):
    """Get binary predictions from pipeline probabilities using given threshold."""
    probs = pipeline.predict_proba(X)[:, 1]
    return (probs >= threshold).astype(int)


def best_threshold(pipeline, X, y):
    """Find the best threshold maximizing Youden's J statistic on given data."""
    y_probs = pipeline.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, y_probs)
    J = tpr - fpr
    idx = np.argmax(J[1:]) + 1 

    return thresholds[idx]

def Youden_index(y_true, y_pred):
    """Compute Youden's J statistic."""
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sensitivity + specificity - 1.0


# optuna related functions

def get_optuna_storage():
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(OPTUNA_STORAGE_PATH),
    )
    return storage


def optimize(study_name, pipe_fn, space_fn, X, y):
    sampler = optuna.samplers.TPESampler(seed=SEED)
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(OPTUNA_STORAGE_PATH),
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=OPTUNA_PRUNER,
        load_if_exists=True
    )

    def objective(trial):
        v_params, m_params, nb_params = space_fn(trial)
        pipe = pipe_fn(v_params, m_params, nb_params)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        scores = []
        for fold_idx, (tr, te) in enumerate(cv.split(X, y), start=1):
            pipe.fit(X.iloc[tr], y.iloc[tr])
            probs = pipe.predict_proba(X.iloc[te])[:, 1]
            scores.append(roc_auc_score(y.iloc[te], probs))

            # report partial mean AUC after each fold and allow pruning
            trial.report(float(np.mean(scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
        return float(np.mean(scores))

    # sequential execution to preserve reproducibility
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS)
    return study


# hyperparameter spaces

def space_tfidf(trial):
    v = {
        "min_df": trial.suggest_int("v_min_df", 1, 3),
        "ngram_range": (1, trial.suggest_int("v_ngram_max", 1, 2))
    }
    m = {
        "solver": "liblinear",
        "C": trial.suggest_float("m_C", 1e-4, 10.0, log=True)
    }
    nb = {}  # unused
    return v, m, nb

def space_nb(trial):
    v = {
        "min_df": trial.suggest_int("v_min_df", 1, 3),
        "ngram_range": (1, trial.suggest_int("v_ngram_max", 1, 2))
    }
    m = {
        "solver": "liblinear",
        "C": trial.suggest_float("m_C", 1e-4, 10.0, log=True)
    }
    nb = {
        "alpha": trial.suggest_float("nb_alpha", 1e-3, 10.0, log=True)
    }
    return v, m, nb


def main():

    # load dataset
    df = load_dataset()
    train_df, dev_df, test_df = split(df)
    X_train, y_train = extract(train_df)
    X_dev, y_dev = extract(dev_df)
    X_test, y_test = extract(test_df)

    # run hyperparameter optimization for TF-IDF + LR pipeline
    study_tfidf = optimize(
        "tfidf_lr",
        lambda v, m, nb=None: tfidf_pipeline(v, m),
        space_tfidf,
        X_train,
        y_train
    )

    # run hyperparameter optimization for NB-weight + LR pipeline
    study_nb = optimize(
        "nbweight_lr",
        lambda v, m, nb: nb_pipeline(v, m, nb),
        space_nb,
        X_train,
        y_train
    )
    print("INFO: Hyperparameter optimization completed.")

    # get the best models with thresholds
    def get_best_model(study, space_fn):
        vp, mp, nbp = space_fn(FixedTrial(study.best_trial.params))
        if study.study_name == "tfidf_lr":
            model = tfidf_pipeline(vp, mp)
        else:
            model = nb_pipeline(vp, mp, nbp)
        model.fit(X_train, y_train)
        # find best threshold on dev set
        best_t = best_threshold(model, X_dev, y_dev)
        return model, best_t
    
    tfidf_model, tfidf_t = get_best_model(study_tfidf, space_tfidf)
    nb_model, nb_t = get_best_model(study_nb, space_nb)

    # evaluate on test set
    y_tfidf_pred = thresholded_predictions(tfidf_model, X_test, tfidf_t)
    y_nb_pred = thresholded_predictions(nb_model, X_test, nb_t)

    # compare models on test set
    from MLstatkit import Delong_test, Bootstrapping

    y_test_probs_tfidf = tfidf_model.predict_proba(X_test)[:, 1]
    y_test_probs_nb = nb_model.predict_proba(X_test)[:, 1]

    # DeLong's test to compare AUCs
    _, p= Delong_test(
        y_test, y_test_probs_tfidf, y_test_probs_nb,
        return_ci=False, return_auc=False,
    )

    # Bootstrap AUCs with 95% CIs
    tfidf_auc, tfidf_auc_cl, tfidf_auc_cu = Bootstrapping(
        y_test, y_test_probs_tfidf, 'roc_auc',
        n_bootstraps=5000, random_state=SEED
    )

    nb_auc, nb_auc_cl, nb_auc_cu = Bootstrapping(
        y_test, y_test_probs_nb, 'roc_auc',
        n_bootstraps=5000, random_state=SEED
    )

    print("INFO: Evaluation metrics computed.")

    # save results

    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "baseline_results.txt", "w") as f:
        f.write("TF-IDF + LR Model:\n")
        f.write(f"Best hyperparameters: {study_tfidf.best_trial.params}\n")
        f.write(f"Best threshold: {tfidf_t:.4f}\n")
        f.write(f"Youden's J statistic: {Youden_index(y_test, y_tfidf_pred):.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, y_tfidf_pred))
        f.write("\n\n")

        f.write("NB-weight + LR Model:\n")
        f.write(f"Best hyperparameters: {study_nb.best_trial.params}\n")
        f.write(f"Best threshold: {nb_t:.4f}\n")
        f.write(f"Youden's J statistic: {Youden_index(y_test, y_nb_pred):.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, y_nb_pred))
        f.write("\n\n")

        f.write("Comparison of TF-IDF and NB-weight models on test set:\n")
        f.write(f"DeLong p-value = {p:.4f}\n")
        f.write(f"TF-IDF AUC = {tfidf_auc:.4f} (95% CI: {tfidf_auc_cl:.4f} - {tfidf_auc_cu:.4f})\n")
        f.write(f"NB-weight AUC = {nb_auc:.4f} (95% CI: {nb_auc_cl:.4f} - {nb_auc_cu:.4f})\n")

    # save confusion matrices
    disp_tfidf = ConfusionMatrixDisplay.from_predictions(y_test, y_tfidf_pred)
    disp_tfidf.figure_.savefig(results_dir / "baseline_tfidf_cm.png")

    disp_nb = ConfusionMatrixDisplay.from_predictions(y_test, y_nb_pred)
    disp_nb.figure_.savefig(results_dir / "baseline_nbweight_cm.png")

    print("INFO: Baseline evaluation completed. Results saved to", results_dir.resolve())

if __name__ == "__main__":
    main()