"""
predict.py — Load saved model artifacts and run cluster + sentiment prediction.

This module assumes the input DataFrame has already been preprocessed by
``src.preprocessing.preprocess_data`` (i.e. it contains a ``new comment``
column).  Call ``preprocess_data`` once in your pipeline *before* calling
``predict_on_new_data`` — do not call it again inside this module.

Typical usage
-------------
>>> from src.preprocessing import preprocess_data
>>> from src.predict import load_model_artifacts, predict_on_new_data
>>>
>>> df = pd.read_csv("input.csv")
>>> df_processed = preprocess_data(df, text_column="comment")
>>> artifacts = load_model_artifacts("models/")
>>> df_result = predict_on_new_data(df_processed, artifacts)
"""

import os
import pickle

import numpy as np
import pandas as pd

from src.sentiment_model import predict_transformer_sentiment


def load_model_artifacts(model_dir: str = "models") -> dict:
    """Load all trained model artefacts from *model_dir*.

    Expected files::

        cluster_vectorizer.pkl
        sentiment_vectorizer.pkl
        cluster_label_encoder.pkl
        sentiment_label_encoder.pkl
        best_cluster_model.pkl
        best_sentiment_model.pkl

    Args:
        model_dir: Directory containing the ``.pkl`` files.

    Returns:
        Dict mapping artefact name (without ``.pkl``) → loaded object.

    Raises:
        FileNotFoundError: If any required ``.pkl`` file is missing.
    """
    artifacts = {}
    for filename in [
        "cluster_vectorizer.pkl",
        "sentiment_vectorizer.pkl",
        "cluster_label_encoder.pkl",
        "sentiment_label_encoder.pkl",
        "best_cluster_model.pkl",
        "best_sentiment_model.pkl",
    ]:
        path = os.path.join(model_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model artefact not found: {path}")
        with open(path, "rb") as f:
            artifacts[filename.replace(".pkl", "")] = pickle.load(f)
    return artifacts


def predict_on_new_data(df: pd.DataFrame, artifacts: dict, skip_transformer_sentiment: bool = False) -> pd.DataFrame:
    """Run cluster and sentiment prediction on an already-preprocessed DataFrame.

    The DataFrame must contain a ``new comment`` column produced by
    ``src.preprocessing.preprocess_data``.

    Predictions added:
    - ``predicted_cluster``              — topic cluster label (str)
    - ``predicted_ml_sentiment``         — sentiment from ML model (str)
    - ``transformer_sentiment``          — sentiment from RuBERT transformer (str)
    - ``transformer_sentiment_confidence`` — confidence score (float)

    Args:
        df:                          Preprocessed DataFrame (must have ``new comment``).
        artifacts:                   Dict returned by :func:`load_model_artifacts`.
        skip_transformer_sentiment:  If ``True``, skip the transformer inference
                                     step (faster but less accurate sentiment).

    Returns:
        Copy of *df* with prediction columns appended.
    """
    df_processed = df.copy()

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
    if not skip_transformer_sentiment:
        df_processed = predict_transformer_sentiment(df_processed, text_column='comment')
    else:
        print("Skipping transformer-based sentiment prediction as requested.")
        df_processed['transformer_sentiment'] = None
        df_processed['transformer_sentiment_confidence'] = None

    return df_processed
