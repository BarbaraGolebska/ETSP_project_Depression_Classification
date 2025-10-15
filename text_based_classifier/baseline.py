import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer


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
    token_pattern = re.compile(r"(?u)\b[^\W\d_]{2,}\b")  # yoinked from TfidfVectorizer
    txt = "" if not isinstance(text, str) else text.lower()
    words = token_pattern.findall(txt)
    pos_tags = pos_tag(words)
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    return lemmatized_words


def main():
    # get the combined dataset
    df = pd.read_csv("../data/raw/text_combined.csv", index_col=0)

    # get only "train" part of the dataset
    df_train = df[df["split"] == "train"]

    # get the necessary columns out of df_train
    train_texts = df_train["text"]
    y_train = df_train["target_depr"]

    vectorizer = TfidfVectorizer(tokenizer=nltk_preprocess)  # todo consider ngrams?
    X_train = vectorizer.fit_transform(train_texts)

    feature_names = vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(X_train.toarray(), columns=feature_names)

    print(tfidf_df)


if __name__ == "__main__":
    main()