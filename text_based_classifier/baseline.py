import pandas as pd
import re

from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from imblearn.pipeline import make_pipeline


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


def main():
    df = load_dataset()

    # get only train part of the dataset
    df_train = get_split(df, "train")
    # get only dev part of the dataset
    df_dev = get_split(df, "dev")

    X_train, y_depr_train, X_dev, y_depr_dev = get_X_y_split(df_train, df_dev)

    vectorizer= TfidfVectorizer(tokenizer=nltk_preprocess,
                    ngram_range=(1, 2),
                    min_df=2,  # ignore words that appear in less than 2 documents
                    token_pattern=None,
                )
    undersampler = RandomUnderSampler(random_state=42)
    lr = LogisticRegression(random_state=42)

    pipeline = make_pipeline(vectorizer, undersampler, lr)
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