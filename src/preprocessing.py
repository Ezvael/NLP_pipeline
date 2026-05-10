import re
from functools import lru_cache

import nltk
import pandas as pd
from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import TweetTokenizer
from pymorphy3 import MorphAnalyzer
from stop_words import get_stop_words

from src.cleaner import clean_text

# Regex patterns for initial clustering
patterns = {
    "chatbot": r"""\b(
(чат[\s\-]?бот\w*)|
(chat[\s\-]?bot\w*)|
(chatbot\b)|
(бот(ы|а|ов|у|ом|ами)?)|
((?:онлайн|ии|виртуал|электрон)[\s\-]?ассист\w*)|
(assist\w*)|
((?:онлайн|ии|виртуал|электрон)[\s\-]?помо[шщ]н\w*)
)""",
    "personalization": r"""\b(
(персонализ\w*)|
(рекомендац\w*)|
(персонифик\w*)|
(индивидуализ\w*)
)""",
    "pricing": r"""\b(
((?:цен|рассчит|прайс|стоимость).[1-15](?:мен[аяи]|формирован|установлен|расс?чет)\w*)|
((?:мен[аяи]|формирован|установлен|расс?чет).*?(?:цен|рассчит|прайс|стоимость))|
(ценообразование\b)|
(прейскурант\b)
)""",
    "suggest": r"""\b(
(саджест\b)|
(подсказ\w*)|
(автозап\w*)|
(автопод\w*)
)""",
    "timing": r"""\b(
(дост.*?врем\w*)|
(врем.*?дост\w*)|
(когда.*?(?:дост|прий?деё?т|заказ)\w*)|
(задерж\w*)|
(ожида\w*)|
(жд[уа]\w*)|
(долго\b)|
(отсроч\w*)|
(затягива\w*)|
(промедлен\w*)
)"""
}

# ── Shared morphological analyser ─────────────────────────────────────────────
morph = MorphAnalyzer()

# ── NEW lemmatizer: LRU-cached + stop_words library ───────────────────────────
stopwords_new = set(get_stop_words("ru"))


@lru_cache(maxsize=100_000)
def _get_lemma(word: str) -> str:
    """Return the normal form of *word* (cached for performance)."""
    return morph.parse(word)[0].normal_form


def lemmatize_new(text: str) -> str:
    """New lemmatizer: clean_text + split + stop_words + LRU-cached pymorphy3.

    Removes URLs, lowercases, discards tokens shorter than 3 chars and
    Russian stopwords (stop_words library), then lemmatises with pymorphy3.
    Best suited for RuBERT fine-tuning or when URL-noise removal matters.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    cleaned = clean_text(text)
    tokens = cleaned.split()
    lemmas = [
        _get_lemma(t) for t in tokens
        if len(t) >= 3 and t not in stopwords_new
    ]
    return " ".join(lemmas)


# ── OLD lemmatizer: TweetTokenizer + NLTK stopwords + isalpha ─────────────────
_tweet_tokenizer = TweetTokenizer()
_stopwords_nltk: set | None = None  # lazy-loaded so NLTK download is optional


def _get_nltk_stopwords() -> set:
    global _stopwords_nltk
    if _stopwords_nltk is None:
        try:
            _stopwords_nltk = set(nltk_stopwords.words("russian"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            _stopwords_nltk = set(nltk_stopwords.words("russian"))
    return _stopwords_nltk


def lemmatize_old(text: str) -> str:
    """Old lemmatizer: TweetTokenizer + NLTK Russian stopwords + isalpha filter.

    Uses TweetTokenizer (handles emoticons/hashtags), discards non-alpha tokens
    and NLTK Russian stopwords, then lemmatises with pymorphy3.
    Historically used for TF-IDF ML models.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    tokens = _tweet_tokenizer.tokenize(text.lower())
    sw = _get_nltk_stopwords()
    lemmas = [
        morph.parse(t)[0].normal_form
        for t in tokens
        if t.isalpha() and t not in sw
    ]
    return " ".join(lemmas)


# Default alias — kept as 'lemmatize' for backwards compatibility
lemmatize = lemmatize_new


# ── Regex multi-label classifier ──────────────────────────────────────────────
def classify_multi(df, text_col, patterns, new_col="clusters"):
    compiled = {k: re.compile(v, re.IGNORECASE | re.VERBOSE) for k, v in patterns.items()}

    def classify(text):
        if pd.isna(text) or not isinstance(text, str):
            return []
        found = []
        for label, pattern in compiled.items():
            if pattern.search(text):
                found.append(label)
        return found

    df[new_col] = df[text_col].apply(classify)
    return df


# ── Preprocessing pipelines ───────────────────────────────────────────────────
def preprocess_data(df, text_column='comment', lemmatizer='new'):
    """Preprocess a DataFrame: regex clustering + lemmatization.

    Args:
        df:           Input DataFrame.
        text_column:  Name of the raw text column (default ``'comment'``).
        lemmatizer:   ``'new'`` (LRU-cached + stop_words, default) or
                      ``'old'`` (TweetTokenizer + NLTK stopwords + isalpha).

    Returns:
        DataFrame with two extra columns:
        - ``clusters``    — list of regex-matched cluster labels
        - ``new comment`` — lemmatized text used by ML models
    """
    df = classify_multi(df, text_column, patterns)
    lem_fn = lemmatize_old if lemmatizer == 'old' else lemmatize_new
    df['new comment'] = df[text_column].astype(str).apply(lem_fn)
    return df
