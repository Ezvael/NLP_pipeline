from src.nlp.cleaner import clean_text
from src.nlp.lemmatizer import Lemmatizer
from src.nlp.feature_extractor import FeatureExtractor
from src.nlp.topic_detector import TopicDetector

class NLPPipeline:

    def __init__(self):

        self.lemmatiser = Lemmatizer()
        self.feature_extractor = FeatureExtractor()
        self.topic_detector = TopicDetector()

    def process(self, raw_text):

        raw_text = str(raw_text)
        cleaned = clean_text(raw_text)
        lemmatized = self.lemmatiser.lemmatize(cleaned)
        lemmas = lemmatized.split()
        tags = self.feature_extractor.extract_tags(cleaned,lemmas)
        topics = self.topic_detector.detect(lemmas)

        return {
            "raw_text": raw_text,
            "clean_text": cleaned,
            "lemmatized_text": lemmatized,
            "tags": tags,
            "topics": topics,
        }