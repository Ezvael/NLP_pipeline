from src.config import TOPIC_KEYWORDS

class TopicDetector:

    def detect(self, tokens):

        token_set = set(tokens)

        return [topic for topic, keywords in TOPIC_KEYWORDS.items() if token_set & keywords]