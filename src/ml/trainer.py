from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

class ModelTrainer:

    def __init__(self):

        self.cluster_encoder = LabelEncoder()
        self.sentiment_encoder = LabelEncoder()
    
    def train_cluster_model(self, X, y):

        y = self.cluster_encoder.fit_transform(y)
        model = LogisticRegression(max_iter=2000, n_jobs=-1)
        model.fit(X, y)
        self.cluster_model = model
    
    def train_sentiment_model(self, X, y):

        y = self.sentiment_encoder.fit_transform(y)
        model = LogisticRegression(max_iter=2000, n_jobs=-1)
        model.fit(X, y)
        self.sentiment_model = model