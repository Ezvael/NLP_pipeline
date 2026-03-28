import re
from functools import lru_cache
from pymorphy3 import MorphAnalyzer
# import from config file
from nlp_config import (
    POS_EMOJI, NEG_EMOJI,
    POS_SLANG, NEG_SLANG,
    PROFANITY, BRANDS,
    SARCASM_MARKERS, NEG_WORDS,
    TOPIC_KEYWORDS
)

morph = MorphAnalyzer()

ELONG_PATTERN = re.compile(r"(.)\1{2,}")

@lru_cache(maxsize=50000)
def get_lemma(word: str) -> str:
    return morph.parse(word)[0].normal_form

def normalize_elongation(word: str):
    if ELONG_PATTERN.search(word):
        return ELONG_PATTERN.sub(r"\1", word), True
    return word, False

LAUGHTER_PATTERN = re.compile(r"\b(а?х[ае]?){2,}|(х[ае]){2,}|хи{2,}", re.I)
URL_PATTERN = re.compile(r"http\S+|www\S+")
EXCL_PATTERN = re.compile(r"!{2,}")
QUES_PATTERN = re.compile(r"\?{2,}")
CYR_PATTERN = re.compile(r"[^а-яё# ]", re.I)

def extract_emoji_tags(text):
    tags = []
    chars = set(text)
    if chars & POS_EMOJI:
        tags.append("TAG_EMO_POS")
    if chars & NEG_EMOJI:
        tags.append("TAG_EMO_NEG")
    return tags

def detect_sarcasm(text):
    if any(m in text for m in SARCASM_MARKERS) and "!" in text:
        return ["TAG_SARCASM"]
    return []

def detect_negation(doc):
    negated = set()
    for token in doc:
        if token.text in NEG_WORDS:
            head = token.head
            if head.is_alpha:
                negated.add(head.i)
    return negated

def detect_topics(tokens):
    token_set = set(tokens)
    return [
        topic for topic, keywords in TOPIC_KEYWORDS.items()
        if token_set & keywords
    ]