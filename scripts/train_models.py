import os
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from src.ml.feature_builder import FeatureBuilder

MODEL_DIR = "models"

def load_training_data():

    df = pd.read_parquet("data/processed/comments_enriched.parquet")

    return df

def train():

    df = load_training_data()
    df = df[df["lemmatized_text"].fillna("").str.len() > 0] # remove empty rows
    cluster_df = df.dropna(subset=["cluster_mode"]) # cluster labels
    sentiment_df = df.dropna(subset=["sentiment_mode"]) # sentiment labels

    # feature_builder
    feature_builder = FeatureBuilder(max_features=30000)
    feature_builder.fit(df["lemmatized_text"])

    # cluster model
    print("Training cluster model...")
    cluster_encoder = LabelEncoder()
    y_cluster = cluster_encoder.fit_transform(cluster_df["cluster_mode"])
    X_cluster = feature_builder.transform(cluster_df)
    cluster_model = LogisticRegression(max_iter=2000, n_jobs=-1)
    cluster_model.fit(X_cluster, y_cluster)
    cluster_preds = cluster_model.predict(X_cluster)
    print(classification_report(y_cluster, cluster_preds))

    # sentiment model
    print("Training sentiment model...")
    sentiment_encoder = LabelEncoder()
    y_sentiment = sentiment_encoder.fit_transform(sentiment_df["sentiment_mode"])
    X_sentiment = feature_builder.transform(sentiment_df)
    sentiment_model = LogisticRegression(max_iter=2000, n_jobs=-1)
    sentiment_model.fit(X_sentiment, y_sentiment)
    sentiment_preds = sentiment_model.predict(X_sentiment)
    print(classification_report(y_sentiment, sentiment_preds))

    # saving models
    os.makedirs(MODEL_DIR, exist_ok=True)

    artifacts = {
        "feature_builder.pkl": (feature_builder),
        "cluster_model.pkl": (cluster_model),
        "sentiment_model.pkl": (sentiment_model),
        "cluster_encoder.pkl": (cluster_encoder),
        "sentiment_encoder.pkl": (sentiment_encoder)
    }

    for filename, obj in (artifacts.items()):

        with open(f"{MODEL_DIR}/{filename}", "wb") as f: pickle.dump(obj,f)

    print("Training complete!")

if __name__ == "__main__": train()