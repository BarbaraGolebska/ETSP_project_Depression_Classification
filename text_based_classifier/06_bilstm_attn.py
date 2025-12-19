from pathlib import Path
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import optuna
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from tqdm import tqdm

# module tokenizer handles downloading of NLTK resources
from tokenizer import nltk_sentence_tokenize as sent_tokenize

RESULTS_DIR = "./results"
EMBEDDINGS_PATH = "../data/processed/bilstm_attn_embeddings.joblib"

OPTUNA_N_TRIALS = 30
OPTUNA_STORAGE_PATH = "./optuna/bilstm_attn_optuna_journal_storage.log"
# deterministic pruner (median over previous trials)
OPTUNA_PRUNER = optuna.pruners.MedianPruner(
    n_warmup_steps=1,   # don't prune before at least 1 reported step
    n_min_trials=8      # wait for at least 8 completed trials
)

# device selection for heavy models
if torch.cuda.is_available(): DEVICE = 'cuda'
elif torch.backends.mps.is_available(): DEVICE = 'mps'
else: DEVICE = 'cpu'
print(f"INFO: Using device: {DEVICE}")


# set random seeds for reproducibility
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == 'cuda':
    torch.cuda.manual_seed_all(SEED)

# embedding dimension
EMB_DIM = 768
# fixed maximum number of training epochs
MAX_EPOCHS = 12


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
    X = np.stack(df["embedding"].values)  # (N, T, D)
    M = np.stack(df["mask"].values)       # (N, T)
    y = df["target_depr"].to_numpy()      # (N,)

    return X, M, y


def get_datasets(df):
    train_df, dev_df, test_df = split(df)
    X_train, M_train, y_train = extract(train_df)
    X_dev, M_dev, y_dev = extract(dev_df)
    X_test, M_test, y_test = extract(test_df)

    train_ds = DocDataset(X_train, M_train, y_train)
    dev_ds = DocDataset(X_dev, M_dev, y_dev)
    test_ds = DocDataset(X_test, M_test, y_test)
    return train_ds, dev_ds, test_ds


def get_dataloaders(train_ds, dev_ds, test_ds, batch_size):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, dev_loader, test_loader


# preprocessing and embedding related functions

def get_punctuation_model():
    print("INFO: Loading punctuation model...")
    from punctuationmodel import PunctuationModel
    return PunctuationModel(device=DEVICE)


def get_embedding_model():
    print("INFO: Loading embedding model...")
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=DEVICE)


def build_embeddings(df, text_col="text", emb_col="embedding", mask_col="mask", max_sentences=128):
    punctuation_model = get_punctuation_model()
    embedding_model = get_embedding_model()
    
    all_embeddings = []
    all_masks = [] # list of masks indicating valid sentences and padded ones

    for _, row in tqdm(df.iterrows(), total=len(df), desc="INFO: Building embeddings"):
        text = str(row[text_col])
        
        punc_text = punctuation_model.restore_punctuation(text)
        sentences = sent_tokenize(punc_text)

        # optional: fiter out short sentences
        # sentences = [s for s in sentences if len(s.split()) >= 3]

        if len(sentences) == 0: 
            sentences = [""]

        # truncate to max number of sentences
        sentences = sentences[:max_sentences]

        embeddings = embedding_model.encode(sentences, convert_to_numpy=True)  # (T, D)

        T, D = embeddings.shape
        # padding if a document has less than max_sentences sentences
        if T < max_sentences:
            pad = np.zeros((max_sentences - T, D), dtype=embeddings.dtype)
            embeddings = np.vstack([embeddings, pad])
            mask = np.zeros(max_sentences, dtype=np.int64)
            mask[:T] = 1 # 1 - valid sentence, 0 - padded
        else:
            mask = np.ones(max_sentences, dtype=np.int64)

        all_embeddings.append(embeddings)
        all_masks.append(mask)

    result_df = df.copy()
    result_df[emb_col] = all_embeddings
    result_df[mask_col] = all_masks
    return result_df


# models

class Attention(nn.Module):
    """
    Attention-based classifier over sentence embeddings.
    """
    def __init__(self, emb_dim, attn_hidden=128, dropout=0.3):
        super().__init__()
        self.attn_proj = nn.Linear(emb_dim, attn_hidden)
        self.attn_v = nn.Linear(attn_hidden, 1, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x, mask):
        # additive attention
        proj = torch.tanh(self.attn_proj(x))      # (B, T, H_attn)
        scores = self.attn_v(proj).squeeze(-1)    # (B, T)

        # padded sentences get very low attention scores
        scores = scores.masked_fill(mask == 0, -1e9)

        # attention weights: sum to 1 over valid sentences, 0 for padded ones
        attn_weights = torch.softmax(scores, dim=1)      # (B, T)
        attn_weights = attn_weights.unsqueeze(-1)        # (B, T, 1)

        # weighted sum of sentence embeddings to get document representation
        doc_repr = torch.sum(x * attn_weights, dim=1)   # (B, D)
        doc_repr = self.dropout(doc_repr)

        # classification
        logits = self.classifier(doc_repr).squeeze(-1)      # (B,)

        return logits, attn_weights.squeeze(-1), doc_repr


class BiLSTMAttn(nn.Module):
    """
    BiLSTM with Attention-based classifier over sentence embeddings.
    """
    def __init__(self, emb_dim=768, hidden_size=128, num_layers=1, dropout=0.3, attn_hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.attn_proj = nn.Linear(2 * hidden_size, attn_hidden)
        self.attn_v = nn.Linear(attn_hidden, 1, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
 
    def forward(self, x, mask):
        # BiLSTM
        lstm_out, _ = self.lstm(x)      # (B, T, 2H)
        lstm_out = self.dropout(lstm_out) # (B, T, 2H)
 
        # additive attention
        proj = torch.tanh(self.attn_proj(lstm_out))      # (B, T, H_attn)
        scores = self.attn_v(proj).squeeze(-1)           # (B, T)
 
        # padded sentences get very low attention scores
        scores = scores.masked_fill(mask == 0, -1e9) # (B, T)
 
        # attention weights: sum to 1 over valid sentences, 0 for padded ones
        attn_weights = torch.softmax(scores, dim=1)      # (B, T)
        attn_weights = attn_weights.unsqueeze(-1)        # (B, T, 1)

        # weighted sum of LSTM outputs to get document representation
        doc_repr = torch.sum(lstm_out * attn_weights, dim=1) # (B, 2H)
 
        # Classification
        logits = self.classifier(doc_repr).squeeze(-1)   # (B,)
        return logits, attn_weights.squeeze(-1), doc_repr


# Dataset class for PyTorch DataLoader

class DocDataset(Dataset):
    def __init__(self, X, M, y):
        """
        X: (N, T_max, D)  - sentence embeddings
        M: (N, T_max)     - mask 1/0 (1 = real sentence, 0 = padded ones)
        y: (N,)           - binary labels (0/1)
        """
        if isinstance(X, torch.Tensor):
            self.X = X.to(torch.float32)
        else:
            self.X = torch.tensor(X, dtype=torch.float32)

        if isinstance(M, torch.Tensor):
            self.M = M.to(torch.long)
        else:
            self.M = torch.tensor(M, dtype=torch.long)

        if isinstance(y, torch.Tensor):
            self.y = y.to(torch.float32)
        else:
            self.y = torch.tensor(y, dtype=torch.float32)  # for BCEWithLogitsLoss

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.M[idx], self.y[idx]


# get class weights for imbalanced datasets

def get_pos_weights(y):
    """Calculate positive class weight for imbalanced binary classification."""
    n_neg = np.sum(y == 0)
    n_pos = np.sum(y == 1)
    pos_weight_value = n_neg / n_pos
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=DEVICE)
    return pos_weight


# optuna related functions

def get_optuna_storage():
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(OPTUNA_STORAGE_PATH),
    )
    return storage


def get_params(trial, model_name):
    # shared hyperparameters
    params = {
        "dropout":      trial.suggest_categorical("dropout", [0.1, 0.3, 0.5]),
        "attn_hidden":  trial.suggest_categorical("attn_hidden", [64, 128]),
        "lr":           trial.suggest_float("lr", 1e-4, 2e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 3e-4, log=True),
        "batch_size":   trial.suggest_categorical("batch_size", [8, 16, 32])
    }

    # model-specific hyperparameters
    if model_name == "BiLSTMAttn":
        params["hidden_size"] = trial.suggest_categorical("hidden_size", [64, 128])
    
    return params


def get_model(model_name, params):
    if model_name == "Attention":
        model = Attention(
            emb_dim=EMB_DIM,
            attn_hidden=params["attn_hidden"],
            dropout=params["dropout"]
        ).to(DEVICE)
    elif model_name == "BiLSTMAttn":
        model = BiLSTMAttn(
            emb_dim=EMB_DIM,
            hidden_size=params["hidden_size"],
            dropout=params["dropout"],
            attn_hidden=params["attn_hidden"]
        ).to(DEVICE)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Supported: Attention, BiLSTMAttn")
    
    return model


def optimize_hyperparameters(X, M, y, model_name):
    sampler = optuna.samplers.TPESampler(seed=SEED) # for reproducibility
    study_name = f"bilstm_attn_{model_name}"
    study = optuna.create_study(
        study_name=study_name,
        storage=get_optuna_storage(),
        direction='maximize',
        sampler=sampler,
        pruner=OPTUNA_PRUNER,
        load_if_exists=True
    )

    def objective(trial):
        params = get_params(trial, model_name)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        scores = []
        for fold_idx, (train_idxs, test_idxs) in enumerate(cv.split(X, y), start=1):
            model = get_model(model_name, params)

            train_loader = DataLoader(
                DocDataset(X[train_idxs], M[train_idxs], y[train_idxs]),
                batch_size=params["batch_size"],
                shuffle=True
            )
            val_loader = DataLoader(
                DocDataset(X[test_idxs], M[test_idxs], y[test_idxs]),
                batch_size=params["batch_size"],
                shuffle=False
            )

            criterion = nn.BCEWithLogitsLoss(pos_weight=get_pos_weights(y[test_idxs]))
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=params["lr"], 
                weight_decay=params["weight_decay"]
            )

            for _ in range(1, MAX_EPOCHS + 1):
                train(model, train_loader, optimizer, criterion)
        
            scores.append(eval_model(model, val_loader))

            # report partial mean AUC after each fold and allow pruning
            trial.report(np.mean(scores), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(scores)

    # sequential execution to preserve reproducibility
    study.optimize(lambda trial: objective(trial), n_trials=OPTUNA_N_TRIALS)
    return study


# training and evaluation functions

def train(model, loader, optimizer, criterion):
    """ Train the model for one epoch. """
    model.train()
    for X_b, M_b, y_b in loader:
        X_b = X_b.to(DEVICE)
        M_b = M_b.to(DEVICE)
        y_b = y_b.to(DEVICE)

        optimizer.zero_grad()
        logits, _, _ = model(X_b, M_b)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def eval_model(model, loader):
    """ Evaluate the model on the given dataset. Returns AUC. """
    model.eval()
    all_probs, all_targets = [], []

    for X_b, M_b, y_b in loader:
        X_b = X_b.to(DEVICE)
        M_b = M_b.to(DEVICE)
        y_b = y_b.to(DEVICE)

        logits, _, _ = model(X_b, M_b)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.append(probs)
        all_targets.append(y_b.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = np.nan
    return auc

# evaluation functions

@torch.no_grad()
def predict_probs(model, loader):
    """Return concatenated probabilities and targets for a loader."""
    model.eval()
    all_probs, all_targets = [], []
    for X_b, M_b, y_b in loader:
        logits, _, _ = model(X_b.to(DEVICE), M_b.to(DEVICE))
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(y_b.numpy())
    return np.concatenate(all_probs), np.concatenate(all_targets)


def best_threshold(model, loader):
    """Find threshold maximizing Youden's J on the given loader."""
    y_probs, y_true = predict_probs(model, loader)
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    valid = (thresholds > 0.1) & (thresholds < 0.9)
    J = tpr[valid] - fpr[valid]
    idx = np.argmax(J)
    return thresholds[valid][idx]


def Youden_index(y_true, y_pred):
    """Compute Youden's J statistic."""
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sensitivity + specificity - 1.0


def evaluate_test(models):
    """Evaluate models on test set and compare them."""
    from MLstatkit import Delong_test, Bootstrapping
    metrics = {}
    for model_name, model_info in models.items():
        model = model_info["model"]
        threshold = model_info["threshold"]
        test_loader = model_info["test_loader"]

        y_probs, y_test = predict_probs(model, test_loader)
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

    with open(results_dir / "bilstm_attn_results.txt", "w") as f:
        for model_name, model_info in models.items():
            f.write(f"{model_name}:\n")
            f.write(f"Best hyperparameters: {model_info['best_params']}\n")
            f.write(f"Best threshold: {model_info['threshold']:.4f}\n")
            f.write(f"Youden's J statistic: {metrics[model_name]['youden_j']:.4f}\n")
            f.write("\nClassification Report:\n")
            f.write(metrics[model_name]['report'])
            f.write("\n\n")

            # save confusion matrices
            disp = metrics[model_name]['confusion_matrix']
            disp.figure_.savefig(results_dir / f"bilstm_attn_{model_name}_cm.png")

            # save models
            joblib.dump(model_info['model'], results_dir / f"bilstm_attn_{model_name}.pkl")

        # save comparison results
        f.write("Comparison of the models on test set:\n")
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
    embeddings_cache = Path(EMBEDDINGS_PATH)
    if embeddings_cache.exists():
        print("INFO: Loading cached embeddings...")
        embeddings_df = joblib.load(embeddings_cache)
    else:
        embeddings_df = build_embeddings(df)
        print("INFO: Embeddings built.")
        embeddings_cache.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(embeddings_df, embeddings_cache)
        print(f"INFO: Embeddings saved to {embeddings_cache.resolve()}.")

    train_ds, dev_ds, test_ds = get_datasets(embeddings_df)
    X_train, M_train, y_train = extract(embeddings_df[embeddings_df.split == "train"])

    # find best models
    model_names = ["Attention", "BiLSTMAttn"]
    models = {}
    for model_name in model_names:

        # run hyperparameter optimization
        study = optimize_hyperparameters(X_train, M_train, y_train, model_name)
        print(f"INFO: Hyperparameter optimization for {model_name} completed.")

        # get the model with the chosen parameters
        best_params = study.best_params
        model = get_model(model_name, best_params)

        # train final model on the full training set
        batch_size = best_params["batch_size"]
        train_loader, dev_loader, test_loader = get_dataloaders(train_ds, dev_ds, test_ds, batch_size)
        criterion = nn.BCEWithLogitsLoss(pos_weight=get_pos_weights(y_train))
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])

        for _ in tqdm(range(1, MAX_EPOCHS + 1), 
                      desc=f"INFO: Training final {model_name} model"):
            train(model, train_loader, optimizer, criterion)
        
        # determine best threshold on dev set
        threshold = best_threshold(model, dev_loader)
        
        models[model_name] = {
            "study": study,
            "model": model,
            "threshold": threshold,
            "best_params": best_params,
            "test_loader": test_loader # for later evaluation
        }

    # evaluate and save results
    print("INFO: Evaluating models on test set...")
    metrics = evaluate_test(models)
    
    # save results
    save_results(models, metrics)        

if __name__ == "__main__":
    main()
