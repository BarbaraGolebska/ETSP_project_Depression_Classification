import numpy as np
import pandas as pd
import re
import optuna

from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, fbeta_score
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold


# NLTK related functions

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
    lr = LogisticRegression(**model_params,
                            random_state=42)

    pipeline = make_pipeline(vectorizer, undersampler, lr)
    return pipeline


def train_evaluate(X_train, y_depr_train, model_params):
    scores = []

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for train_index, test_index in cv.split(X_train, y_depr_train):
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_depr_train.iloc[train_index], y_depr_train.iloc[test_index]

        pipeline = get_pipeline(model_params)
        pipeline.fit(X_train_fold, y_train_fold)

        predictions = pipeline.predict(X_test_fold)
        scores.append(fbeta_score(y_test_fold, predictions, beta=2))  # recall twice as important as precision

    return np.mean(scores)


def objective(trial, X_train, y_depr_train):
    model_params = {'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'newton-cg']),  # small dataset friendly
              'C': trial.suggest_float("C", 1e-7, 10.0, log=True)}

    return train_evaluate(X_train, y_depr_train, model_params)


def main():
    df = load_dataset()

    # get only train part of the dataset
    df_train = get_split(df, "train")
    # get only dev part of the dataset
    df_dev = get_split(df, "dev")

    X_train, y_depr_train, X_dev, y_depr_dev = get_X_y_split(df_train, df_dev)

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_depr_train), n_trials=100)

    # get the pipeline wth the chosen parameters
    pipeline = get_pipeline(study.best_trial.params)

    pipeline.fit(X_train, y_depr_train)

    # evaluate on dev set
    y_depr_pred = pipeline.predict(X_dev)

    report_dict = classification_report(y_depr_dev, y_depr_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()

    print("Classification report on dev set:")
    print(report_df)

    # print confusion matrix
    ConfusionMatrixDisplay.from_estimator(pipeline, X_dev, y_depr_dev)
    plt.show()


if __name__ == "__main__":
    main()