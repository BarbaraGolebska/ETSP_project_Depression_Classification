from pathlib import Path
import joblib
import numpy as np
import torch
import optuna
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.pipeline import make_pipeline
from tqdm import tqdm

# module tokenizer handles downloading of NLTK resources
from tokenizer import nltk_sentence_tokenize as sent_tokenize

tqdm.pandas() # add progress_apply to pandas to show progress bars

RESULTS_DIR = "./results"

OPTUNA_N_TRIALS = 1000
OPTUNA_STORAGE_PATH = "./embeddings-based_optuna_journal_storage.log"
# deterministic pruner (median over previous trials)
OPTUNA_PRUNER = optuna.pruners.MedianPruner(
    n_warmup_steps=1,   # don't prune before at least 1 reported step
    n_min_trials=8      # wait for at least 8 completed trials
)

# device selection for heavy models
if torch.cuda.is_available(): DEVICE = 'cuda'
elif torch.backends.mps.is_available(): DEVICE = 'mps'
else: DEVICE = 'cpu'

# set random seeds for reproducibility
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

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
    return np.vstack(df["embedding"].to_numpy()), df["target_depr"].astype(int).to_numpy()


# preprocessing and embedding related functions

def get_punctuation_model():
    print("INFO: Loading punctuation model...")
    from punctuationmodel import PunctuationModel
    return PunctuationModel(device=DEVICE)


def get_embedding_model():
    print("INFO: Loading embedding model...")
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=DEVICE)


def build_embeddings(df, text_col="text", emb_col="embedding"):
    tqdm.pandas(desc="INFO: Building embeddings")
    punctuation_model = get_punctuation_model()
    embedding_model = get_embedding_model()
    @torch.inference_mode() # turn off gradients to speed up inference
    def _preprocess_text(text):
        punc_text = punctuation_model.restore_punctuation(text)
        sentences = sent_tokenize(punc_text)
        embeddings = embedding_model.encode(sentences, convert_to_tensor=True)
        return embeddings.mean(dim=0).cpu().numpy() # mean pooling

    result_df = df.copy()
    result_df[emb_col] = result_df[text_col].astype(str).progress_apply(lambda s: _preprocess_text(s))

    return result_df


# pipeline

def get_pipeline(model_params, model_name):
    undersampler = RandomUnderSampler(random_state=SEED)

    if model_name=="CatBoost":
        model = CatBoostClassifier(verbose=0, random_state=SEED, **model_params)
    elif model_name=="MLP":
        model = MLPClassifier(random_state=SEED, max_iter=3000, **model_params)
    elif model_name=="LR":
        model = LogisticRegression(random_state=SEED, **model_params)

    return make_pipeline(undersampler, model)


# optuna related functions

def get_optuna_storage():
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(OPTUNA_STORAGE_PATH),
    )
    return storage


def get_model_params(trial, model_name):
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
            model_params["subsample"] = trial.suggest_float("subsample", 0.3, 1)
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
    elif model_name == "LR":
        model_params = {
            'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'newton-cg']),
            'C': trial.suggest_float("C", 1e-7, 10.0, log=True)
        }

    # store the cleaned params on the trial
    trial.set_user_attr("model_params", model_params)

    return model_params


def optimize_hyperparameters(X, y, model_name):
    sampler = optuna.samplers.TPESampler(seed=SEED) # for reproducibility
    study_name = f"embeddings-based_{model_name}"
    study = optuna.create_study(
        study_name=study_name,
        storage=get_optuna_storage(),
        direction='maximize',
        sampler=sampler,
        pruner=OPTUNA_PRUNER,
        load_if_exists=True
    )

    def objective(trial):
        model_params = get_model_params(trial, model_name)
        pipe = get_pipeline(model_params, model_name)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        scores = []
        for fold_idx, (train_idxs, test_idxs) in enumerate(cv.split(X, y), start=1):
            pipe.fit(X[train_idxs], y[train_idxs])
            probs = pipe.predict_proba(X[test_idxs])[:, 1]
            scores.append(roc_auc_score(y[test_idxs], probs))

            # report partial mean AUC after each fold and allow pruning
            trial.report(np.mean(scores), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
   
        return np.mean(scores)

    # sequential execution to preserve reproducibility
    study.optimize(lambda trial: objective(trial), n_trials=OPTUNA_N_TRIALS)
    return study


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
    idx = np.argmax(J)

    return thresholds[idx]

def Youden_index(y_true, y_pred):
    """Compute Youden's J statistic."""
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sensitivity + specificity - 1.0


def evaluate_models(models, X_test, y_test):
    """Evaluate models on test set and compare them."""
    from MLstatkit import Delong_test, Bootstrapping
    metrics = {}
    for model_name, model_info in models.items():
        pipeline = model_info["pipeline"]
        threshold = model_info["threshold"]

        y_probs = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_probs >= threshold).astype(int)

        youden_j = Youden_index(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        # bootstrap AUCs with 95% CIs
        auc, auc_cl, auc_cu = Bootstrapping(
            y_test, y_probs, 'roc_auc',
            n_bootstraps=5000, random_state=SEED
        )

        # confusion matrix
        disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

        metrics[model_name] = {
            "youden_j": youden_j,
            "report": report,
            "y_probs": y_probs,
            "auc": auc,
            "auc_cl": auc_cl,
            "auc_cu": auc_cu,
            "confusion_matrix": disp
        }

    # DeLong's test to compare AUCs
    model_names = list(models.keys())
    _, p= Delong_test(
        y_test, metrics[model_names[0]]["y_probs"],
        metrics[model_names[1]]["y_probs"],
        return_ci=False, return_auc=False,
    )
    metrics["delong_p_value"] = p
    return metrics


def save_results(models, metrics):
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "embeddings_results.txt", "w") as f:
        for model_name, model_info in models.items():
            f.write(f"Embeddings + {model_name}:\n")
            f.write(f"Best hyperparameters: {model_info['study'].best_trial.params}\n")
            f.write(f"Best threshold: {model_info['threshold']:.4f}\n")
            f.write(f"Youden's J statistic: {metrics[model_name]['youden_j']:.4f}\n")
            f.write("\nClassification Report:\n")
            f.write(metrics[model_name]['report'])
            f.write("\n\n")

            # save confusion matrices
            disp = metrics[model_name]['confusion_matrix']
            disp.figure_.savefig(results_dir / f"embeddings-based_{model_name}_cm.png")

            # save models
            joblib.dump(model_info['pipeline'][1], results_dir / f"embeddings-based_{model_name}.pkl")

        # save comparison results
        f.write("Comparison of embeddings-based models on test set:\n")
        f.write(f"DeLong p-value = {metrics['delong_p_value']:.4f}\n")
        for model_name in models.keys():
            auc = metrics[model_name]['auc']
            auc_cl = metrics[model_name]['auc_cl']
            auc_cu = metrics[model_name]['auc_cu']
            f.write(f"{model_name} AUC = {auc:.4f} (95% CI: {auc_cl:.4f} - {auc_cu:.4f})\n")
    print(f"INFO: Results saved to {results_dir.resolve()}.")


# main function

def main():
    df = load_dataset()
    embeddings_df = build_embeddings(df, text_col="text", emb_col="embedding")
    print("INFO: Embeddings built.")

    train_df, dev_df, test_df = split(embeddings_df)
    X_train, y_train = extract(train_df)
    X_dev, y_dev = extract(dev_df)
    X_test, y_test = extract(test_df)

    # find best models
    model_names = ["MLP", "CatBoost", "LR"]
    models = {}
    for model_name in model_names:

        # run hyperparameter optimization
        study = optimize_hyperparameters(X_train, y_train, model_name)
        print(f"INFO: Hyperparameter optimization for {model_name} completed.")

        # get the pipeline wth the chosen parameters
        best_params = study.best_trial.user_attrs["model_params"]
        pipeline = get_pipeline(best_params, model_name)

        # fit on the full training set and find best threshold on dev set
        pipeline.fit(X_train, y_train)
        threshold = best_threshold(pipeline, X_dev, y_dev)

        models[model_name] = {
            "study": study,
            "pipeline": pipeline,
            "threshold": threshold,
            "best_params": best_params,
        }
        
    # evaluate and save results
    print("INFO: Evaluating models on test set...")
    metrics = evaluate_models(models, X_test, y_test)
    
    # save results
    save_results(models, metrics)

if __name__ == "__main__":
    main()
