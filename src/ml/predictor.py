import pandas as pd

class Predictor:

    def __init__(self, feature_builder, cluster_model, sentiment_model, cluster_encoder, sentiment_encoder):

        self.feature_builder = feature_builder
        self.cluster_model = cluster_model
        self.sentiment_model = sentiment_model
        self.cluster_encoder = cluster_encoder
        self.sentiment_encoder = sentiment_encoder
    
    def predict(self, df):

        X = self.feature_builder.transform(df)
        
        # Cluster prediction
        cluster_preds = self.cluster_model.predict(X)
        cluster_preds = self.cluster_encoder.inverse_transform(cluster_preds)
        df["ml_cluster"] = cluster_preds

        # Sentiment prediction
        sentiment_preds = self.sentiment_model.predict(X)
        sentiment_preds = self.sentiment_encoder.inverse_transform(sentiment_preds)
        df["ml_sentiment"] = sentiment_preds

        return df