"""
feature_extractor.py — Tag-based feature extraction for marketplace comments.

Detects positive/negative emoji, profanity, and slang in raw text and lemmas.
Tags produced here are consumed by ``src.sentiment_model.predict_transformer_sentiment``
to apply a small boost/penalty on top of the RuBERT probability distribution.

Typical usage
-------------
>>> from src.feature_extractor import FeatureExtractor
>>> fe = FeatureExtractor()
>>> tags = fe.extract_tags("Доставка ужасная 😡 блин!", ["доставка", "ужасная", "блин"])
>>> # → ['TAG_EMO_NEG', 'TAG_PROFANITY', 'TAG_SLANG_NEG']
"""

from src.config import NEG_EMOJI, NEG_SLANG, POS_EMOJI, POS_SLANG, PROFANITY


class FeatureExtractor:
    """Extracts a deduplicated list of string tags from a comment."""

    def extract_tags(self, text: str, lemmas: list) -> list:
        """Return a list of tag strings for *text* / *lemmas*.

        Args:
            text:   Raw (or cleaned) comment string — scanned for emoji.
            lemmas: List of lemmatised tokens from the same text — scanned for slang/profanity.

        Returns:
            Deduplicated list of tags from the set:
            ``TAG_EMO_POS``, ``TAG_EMO_NEG``,
            ``TAG_SLANG_POS``, ``TAG_SLANG_NEG``,
            ``TAG_PROFANITY``.
        """
        tags = []
        chars = set(text)

        if chars & POS_EMOJI:
            tags.append("TAG_EMO_POS")

        if chars & NEG_EMOJI:
            tags.append("TAG_EMO_NEG")

        for lemma in lemmas:
            if lemma in PROFANITY:
                tags.append("TAG_PROFANITY")
            if lemma in POS_SLANG:
                tags.append("TAG_SLANG_POS")
            if lemma in NEG_SLANG:
                tags.append("TAG_SLANG_NEG")

        return list(set(tags))
