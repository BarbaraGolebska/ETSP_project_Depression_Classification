from __future__ import annotations
from pathlib import Path
import re
import logging

import nltk
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize

__all__ = ["init_nltk", "nltk_preprocess", "nltk_sentence_tokenize"]

_DEFAULT_NLTK_DIR = Path("../data/nltk_data")
_TAGGER_CANDIDATES = ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger")
_REQUIRED_CORPORA = {
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
}
_READY = False # whether NLTK is initialized
_log = logging.getLogger("tokenizer")


def init_nltk(nltk_dir = _DEFAULT_NLTK_DIR):
    """
    Initialize NLTK data directory and ensure required resources are available.
    """
    global _READY
    if _READY:
        return

    data_dir = Path(nltk_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if str(data_dir) not in nltk.data.path:
        nltk.data.path.append(str(data_dir))

    # Ensure required corpora are downloaded
    for pkg, locator in _REQUIRED_CORPORA.items():
        _ensure_resource(pkg, locator, data_dir)

    # POS tagger (one of the options)
    _ensure_any_tagger(data_dir)

    _READY = True


def _ensure_resource(pkg, locator, data_dir):
    try:
        nltk.data.find(locator)
    except LookupError:
        _log.info("Downloading NLTK resource: %s", pkg)
        nltk.download(pkg, download_dir=str(data_dir), quiet=True)


def _ensure_any_tagger(data_dir):
    for candidate in _TAGGER_CANDIDATES:
        try:
            nltk.data.find(f"taggers/{candidate}")
            return
        except LookupError:
            continue
    nltk.download(_TAGGER_CANDIDATES[0], download_dir=str(data_dir), quiet=True)


def _get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def nltk_preprocess(text, nltk_dir = _DEFAULT_NLTK_DIR):
    """
    Preprocess text using NLTK: tokenize, POS-tag, lemmatize.
    :param text: input text
    :param nltk_dir: NLTK data directory
    :return: list of lemmatized tokens
    """
    if not isinstance(text, str) or not text:
        return []

    if not _READY:
        init_nltk(nltk_dir)

    token_pattern = re.compile(r"[a-z][a-z'-]{1,}") # at least 2 letters, may include apostrophes/hyphens
    words = token_pattern.findall(text.lower())
    if not words:
        return []

    tags = pos_tag(words)
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w, _get_wordnet_pos(t)) for w, t in tags]


def nltk_sentence_tokenize(text, nltk_dir = _DEFAULT_NLTK_DIR):
    """
    Tokenize text into sentences using NLTK.
    :param text: input text
    :param nltk_dir: NLTK data directory
    :return: list of sentences
    """
    if not isinstance(text, str) or not text:
        return []

    if not _READY:
        init_nltk(nltk_dir)

    return sent_tokenize(text)
