import re
from functools import lru_cache

import nltk
import pandas as pd
from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import TweetTokenizer
from pymorphy3 import MorphAnalyzer
from stop_words import get_stop_words

from src.cleaner import clean_text

# Regex patterns for initial cluster pre-filtering.
# Calibrated against a manually annotated sample; canonical version lives in
# the _FILTERS dict inside dev/compare_llm_labelers.py.
patterns = {
    "chatbot": re.compile(
        r"чат.?бот|chat.?bot|бот.{0,15}(не отвеча|перенаправ|завис|цикл|повтор|шаблон)"
        r"|застря[лк].*бот|бот.*застря[лк]"
        r"|автоответ|робот отвеча|оператор недоступен|не могу дозвониться"
        r"|живой.{0,10}(оператор|человек)|переключ.{0,20}оператор",
        re.IGNORECASE,
    ),
    "delay": re.compile(
        r"доставк[аиу]|перенес|опоздал|не приехал|трек|отслежи|курьер|задержк"
        r"|не пришл|жду.*день|перенос.*дат|дата.*изменил"
        r"|не доставил|дата доставки|перенос.*заказ|заказ.*перенес",
        re.IGNORECASE,
    ),
    "pricing": re.compile(
        r"цена.*измени|цена.*поднял|цена.*выросл|стоимость.*измени"
        r"|подорожал|вздорожал|вчера стоил|утром.*цена|цена.*вечером"
        r"|несколько раз.*цен|цену.*подняли|ценообразовани"
        r"|цена.*стала.*дороже|стало.*дороже|было.*рублей.*стало"
        r"|скидка.{0,20}(убра|исчезл|пропал|удали|отмени)"
        r"|цена.{0,10}(поднялась|взлетела|скачет|прыгает|меняется)"
        r"|кэш.?бек.{0,20}(убра|исчезл|снизил|урезал|отмени)"
        r"|повысил.{0,10}цен|снял.{0,10}скидк",
        re.IGNORECASE,
    ),
    "recommendations": re.compile(
        r"рекоменда[цц].{0,30}(не работ|нерелевант|не те|плох|мусор|реклам|алгоритм)"
        r"|алгоритм.{0,30}(рекоменда|подбор|показ|выдач)"
        r"|нерелевантн.{0,30}(товар|реклам|подбор|рекоменда)"
        r"|реклам.{0,20}смешан.{0,20}(товар|рекоменда|подбор)"
        r"|\"для вас\"|похожие товары|вы смотрели|персонализац"
        r"|подборк[аи].{0,30}(реклам|нерелевант|не те|алгоритм|бесполезн)",
        re.IGNORECASE,
    ),
}

# ---------------------------------------------------------------------------
# English cluster patterns
# ---------------------------------------------------------------------------

patterns_en = {
    # ONLY about the support BOT itself (automated agent), NOT human operators.
    "chatbot": re.compile(
        r"chat.?bot|\bbot\b|virtual assist|automat(ed|ic).{0,12}(repl|respon|messag|answer|system)"
        r"|auto.?(repl|respon|answer)|canned (repl|respon|messag|answer)"
        r"|robot.{0,10}(repl|respon|answer)|talking to a (bot|machine|robot)"
        r"|\bAI\b.{0,15}(repl|respon|chat|assist|agent|support)",
        re.IGNORECASE,
    ),
    # Late / missing delivery, stuck tracking, waiting for a specific order.
    "delay": re.compile(
        r"deliver(y|ed|ing)?|shipment|shipping|dispatch|courier|driver"
        r"|track(ing)?|parcel|\bpackage\b"
        r"|(hasn.t|haven.t|not|never|still not|yet to).{0,20}(arriv|deliver|receiv|show|come|turn up)"
        r"|where.{0,12}(my )?(order|package|parcel|item)"
        r"|(late|delayed|overdue|missing|lost|stuck).{0,15}(deliver|order|package|parcel|shipment|item)"
        r"|(order|package|parcel|shipment|item).{0,15}(late|delayed|overdue|missing|lost|stuck)"
        r"|(still |been )?wait(ing|ed).{0,20}(order|package|deliver|parcel|item|days?|weeks?)"
        r"|(prime|2.day|next.?day|same.?day|overnight).{0,20}(late|delay|not (here|arriv|deliver)|still wait)",
        re.IGNORECASE,
    ),
    # ONLY: same item's price changed within a short period.
    "pricing": re.compile(
        r"price.{0,20}(chang|went up|gone up|increas|jump|spike|hike|rais|drop|fell|fluctuat|vary|varies|differ)"
        r"|(chang|went up|increas|jump|spike|hike|drop).{0,12}price"
        r"|(was|were).{0,12}\$?\d.{0,15}(now|today).{0,12}\$?\d"
        r"|(this morning|yesterday|last night|hour ago|day ago|earlier).{0,20}price"
        r"|price.{0,20}(this morning|yesterday|last night|hour ago|day ago|earlier)"
        r"|price.{0,12}(\d|several|multiple|few|keeps?).{0,8}(times?|chang|going up)"
        r"|cart.{0,20}(different|higher|changed|wrong) price"
        r"|price.{0,15}(checkout|cart).{0,15}(different|higher|changed|wrong)",
        re.IGNORECASE,
    ),
    # Recommendation / suggestion algorithm quality — both negative and positive.
    "recommendations": re.compile(
        r"algorithm|personali[zs]"
        r"|recommendation (engine|algorithm|system|feature|page|feed|tab)"
        r"|(suggested|recommended) (for you|products?|items?|listing)"
        r"|\bsuggested for you\b|\bpicks for you\b"
        r"|(recommend|suggest|picks?).{0,25}(irrelevant|random|garbage|useless|terrible|wrong|spam|\bads?\b|same|already (bought|purchas|own))"
        r"|(irrelevant|random|garbage|useless|terrible|wrong|already (bought|purchas|own)).{0,20}(recommend|suggest|picks?)"
        r"|(love|spot on|perfect|amazing|impressed|on point|nailed).{0,20}(recommend|suggest|picks?)"
        r"|(recommend|suggest|picks?).{0,20}(spot on|perfect|amazing|on point|always right|love)"
        r"|why.{0,15}(recommend|suggest).{0,20}(this|that|me|already|same)",
        re.IGNORECASE,
    ),
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
    # Accept both pre-compiled patterns (re.Pattern) and raw regex strings.
    compiled = {
        k: (v if hasattr(v, "search") else re.compile(v, re.IGNORECASE | re.VERBOSE))
        for k, v in patterns.items()
    }

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
