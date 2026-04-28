from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureBuilder:

    def __init__(self):

        self.vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=3, max_df=0.95)
    
    def fit_transform(self, df):

        return self.vectorizer.fit_transform(df["lemmatized_text"])

    def transform(self, df):

        return self.vectorizer.transform(df["lemmatized_text"])