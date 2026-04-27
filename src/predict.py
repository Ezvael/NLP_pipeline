import pandas as pd
import numpy as np
import pickle
import os
import ast
from src.preprocessing import preprocess_data
from src.sentiment_model import predict_transformer_sentiment

def load_model_artifacts(model_dir="models"):
    artifacts = {}
    for filename in ['cluster_vectorizer.pkl', 'sentiment_vectorizer.pkl',
                     'cluster_label_encoder.pkl', 'sentiment_label_encoder.pkl',
                     'best_cluster_model.pkl', 'best_sentiment_model.pkl']:
        with open(os.path.join(model_dir, filename), 'rb') as f:
            artifacts[filename.replace('.pkl', '')] = pickle.load(f)
    return artifacts

def predict_on_new_data(df, artifacts):
    # Preprocess the new data
    df_processed = preprocess_data(df.copy(), text_column='Комментарий')

    # Handle cases where 'new comment' might be empty after preprocessing
    df_processed['new comment'] = df_processed['new comment'].replace('', np.nan).fillna("")

    # Predict Clusters using ML model
    X_new_cluster = artifacts['cluster_vectorizer'].transform(df_processed['new comment'])
    df_processed['predicted_cluster_encoded'] = artifacts['best_cluster_model'].predict(X_new_cluster)
    df_processed['predicted_cluster'] = artifacts['cluster_label_encoder'].inverse_transform(df_processed['predicted_cluster_encoded'])

    # Predict Sentiment using ML model
    X_new_sentiment = artifacts['sentiment_vectorizer'].transform(df_processed['new comment'])
    df_processed['predicted_ml_sentiment_encoded'] = artifacts['best_sentiment_model'].predict(X_new_sentiment)
    df_processed['predicted_ml_sentiment'] = artifacts['sentiment_label_encoder'].inverse_transform(df_processed['predicted_ml_sentiment_encoded'])

    # Predict Sentiment using Transformer model
    df_processed['tags'] = ''
    df_processed = predict_transformer_sentiment(df_processed, text_column='new comment', tags_column='tags')

    return df_processed
