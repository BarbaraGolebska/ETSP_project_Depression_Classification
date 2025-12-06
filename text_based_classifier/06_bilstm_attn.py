from pathlib import Path
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import optuna
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, ConfusionMatrixDisplay, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.pipeline import make_pipeline
from tqdm import tqdm

# module tokenizer handles downloading of NLTK resources
from tokenizer import nltk_sentence_tokenize as sent_tokenize

RESULTS_DIR = "./results"

OPTUNA_N_TRIALS = 50
OPTUNA_STORAGE_PATH = "./bilstm_attn_optuna_journal_storage.log"
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

# embedding dimension
EMB_DIM = 768


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

    # undersample train to balance classes
    rus = RandomUnderSampler(random_state=SEED)
    idx_resampled, y_train = rus.fit_resample(
        np.arange(len(y_train)).reshape(-1, 1),
        y_train
    )
    idx_resampled = idx_resampled.ravel()
    X_train = X_train[idx_resampled]
    M_train = M_train[idx_resampled]

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
        self.X = torch.tensor(X, dtype=torch.float32)
        self.M = torch.tensor(M, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32) # for BCEWithLogitsLoss

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.M[idx], self.y[idx]


# get class weights for imbalanced datasets

def get_pos_weights(y):
    """Calculate positive class weight for imbalanced binary classification."""
    y_np = y.numpy()
    n_neg = np.sum(y_np == 0)
    n_pos = np.sum(y_np == 1)
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
        "dropout":      trial.suggest_float("dropout", 0.1, 0.5),
        "attn_hidden":  trial.suggest_categorical("attn_hidden", [64, 128, 256]),
        "lr":           trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size":   trial.suggest_categorical("batch_size", [8, 16, 32]),
        "max_epochs":   trial.suggest_int("max_epochs", 8, 20),
    }

    # model-specific hyperparameters
    if model_name == "BiLSTMAttn":
        params["hidden_size"] = trial.suggest_categorical("hidden_size", [64, 128, 256])
    
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


def optimize_hyperparameters(train_ds, dev_ds, model_name):
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
        model = get_model(model_name, params)

        # data loaders
        train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
        dev_loader = DataLoader(dev_ds, batch_size=params["batch_size"], shuffle=False)

        # loss function and optimizer
        criterion = nn.BCEWithLogitsLoss() #pos_weight=get_pos_weights(train_ds.y))
        optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

        best_dev_auc = 0.0
        for epoch in range(1, params["max_epochs"] + 1):
            train_loss = train(model, train_loader, optimizer, criterion)
            dev_loss, dev_auc = eval_model(model, dev_loader, criterion)

            # report intermediate objective value to optuna for pruning
            trial.report(dev_auc, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if dev_auc > best_dev_auc:
                best_dev_auc = dev_auc

        return best_dev_auc

    # sequential execution to preserve reproducibility
    study.optimize(lambda trial: objective(trial), n_trials=OPTUNA_N_TRIALS)
    return study


# training and evaluation functions

def train(model, loader, optimizer, criterion):
    """ Train the model for one epoch. """
    model.train()
    total_loss, total_examples = 0.0, 0
    for X_b, M_b, y_b in loader:
        X_b = X_b.to(DEVICE)
        M_b = M_b.to(DEVICE)
        y_b = y_b.to(DEVICE)

        optimizer.zero_grad()
        logits, _, _ = model(X_b, M_b)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_b.size(0)
        total_examples += X_b.size(0)
    return total_loss / total_examples


@torch.no_grad()
def eval_model(model, loader, criterion):
    """ Evaluate the model on the given dataset. Returns loss and AUC. """
    model.eval()
    total_loss, total_examples = 0.0, 0
    all_probs, all_targets = [], []

    for X_b, M_b, y_b in loader:
        X_b = X_b.to(DEVICE)
        M_b = M_b.to(DEVICE)
        y_b = y_b.to(DEVICE)

        logits, _, _ = model(X_b, M_b)
        loss = criterion(logits, y_b)

        total_loss += loss.item() * X_b.size(0)
        total_examples += X_b.size(0)

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(y_b.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = np.nan
    return total_loss / total_examples, auc

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


# main function

def main():
    df = load_dataset()
    embeddings_cache = Path("../data/processed/bilstm_attn_embeddings.joblib")
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

    # find best models
    model_names = ["Attention", "BiLSTMAttn"]
    models = {}
    for model_name in model_names:

        # run hyperparameter optimization
        study = optimize_hyperparameters(train_ds, dev_ds, model_name)
        print(f"INFO: Hyperparameter optimization for {model_name} completed.")

        # get the model wth the chosen parameters
        best_params = study.best_params
        model = get_model(model_name, best_params)

        # train final model on on the full training set and find best threshold on dev set
        print(f"INFO: Training final {model_name} model with best hyperparameters...")

        batch_size = best_params["batch_size"]
        train_loader, dev_loader, test_loader = get_dataloaders(train_ds, dev_ds, test_ds, batch_size)
        criterion = nn.BCEWithLogitsLoss() # pos_weight=get_pos_weights(train_ds.y))
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])

        for epoch in range(1, best_params["max_epochs"] + 1):
            loss = train(model, train_loader, optimizer, criterion)
            print(f"INFO: Epoch {epoch}/{best_params['max_epochs']}, Loss: {loss:.4f}")
        
        threshold = best_threshold(model, dev_loader)
        print(f"INFO: Best threshold on dev set for {model_name}: {threshold:.4f}")

        # print Youden and confusion matrix on dev set
        y_dev_probs, y_dev_true = predict_probs(model, dev_loader)
        y_dev_pred = (y_dev_probs >= threshold).astype(int)
        youden = Youden_index(y_dev_true, y_dev_pred)
        print(f"INFO: Youden's J on dev set for {model_name}: {youden:.4f}")
        print(f"INFO: Confusion Matrix on dev set for {model_name}:")
        print(confusion_matrix(y_dev_true, y_dev_pred))
        print("INFO: Classification Report on dev set:")
        print(classification_report(y_dev_true, y_dev_pred))

        models[model_name] = {
            "study": study,
            "model": model,
            "threshold": threshold,
            "best_params": best_params,
        }
        

if __name__ == "__main__":
    main()
