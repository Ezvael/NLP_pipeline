"""
cleaner.py — Lightweight text cleaning utilities.

Removes URLs, lowercases, and collapses whitespace.
Used by preprocessing.py before lemmatisation.
"""

import re

URL_PATTERN = re.compile(r"http\S+|www\S+")
MULTISPACE   = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Return a cleaned version of *text*: lowercased, URLs stripped, whitespace normalised.

    Args:
        text: Raw input string.

    Returns:
        Cleaned string.
    """
    text = str(text).lower()
    text = URL_PATTERN.sub(" ", text)
    text = MULTISPACE.sub(" ", text)
    return text.strip()
