"""
topic_detector.py — Token-based topic detection for marketplace comments.

Uses lemmatised tokens and the TOPIC_KEYWORDS vocabulary from ``src.config``
to identify which discussion topics a comment belongs to.

This is complementary to the regex-based ``classify_multi`` in preprocessing.py:
- Regex approach: fast pre-filter on raw text (used in overnight_labeling.py).
- TopicDetector: richer token-level matching on lemmatised text (used after NLP pipeline).

Typical usage
-------------
>>> from src.topic_detector import TopicDetector
>>> td = TopicDetector()
>>> td.detect(["доставка", "задержка", "долго"])
>>> # → ['TOPIC_DELIVERY_TIME']
"""

from src.config import TOPIC_KEYWORDS


class TopicDetector:
    """Detects discussion topics from a list of lemmatised tokens."""

    def detect(self, tokens: list) -> list:
        """Return a list of topic names whose keyword set intersects *tokens*.

        Args:
            tokens: List of lemmatised token strings.

        Returns:
            List of matching topic keys (e.g. ``["TOPIC_DELIVERY_TIME"]``).
            Empty list if no topic matches.
        """
        token_set = set(tokens)
        return [
            topic
            for topic, keywords in TOPIC_KEYWORDS.items()
            if token_set & keywords
        ]
