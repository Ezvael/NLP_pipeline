import glob
import pickle
import pandas as pd
from src.nlp.pipeline import NLPPipeline
from src.nlp.sentiment_analyzer import SentimentAnalyzer
from src.ml.predictor import Predictor

MODEL_DIR = "models"
pipeline = NLPPipeline()
transformer_sentiment = SentimentAnalyzer(batch_size=16)

def load_pickle(path):

    with open(path, "rb") as f: return pickle.load(f)

models = {
    "feature_builder": load_pickle(f"{MODEL_DIR}/feature_builder.pkl"),
    "cluster_model": load_pickle(f"{MODEL_DIR}/cluster_model.pkl"),
    "sentiment_model": load_pickle(f"{MODEL_DIR}/sentiment_model.pkl"),
    "cluster_encoder": load_pickle(f"{MODEL_DIR}/cluster_encoder.pkl"),
    "sentiment_encoder": load_pickle(f"{MODEL_DIR}/sentiment_encoder.pkl")
}

predictor = Predictor(
    feature_builder=models["feature_builder"],
    cluster_model=models["cluster_model"],
    sentiment_model=models["sentiment_model"],
    cluster_encoder=models["cluster_encoder"],
    sentiment_encoder=models["sentiment_encoder"]
)

def process_dataframe(df):

    results = []

    for text in df["Комментарий"].fillna(""):

        results.append(pipeline.process(text))

    features_df = pd.DataFrame(results)
    df = pd.concat([df.reset_index(drop=True), features_df], axis=1)

    transformer_df = transformer_sentiment.predict(df["raw_text"].tolist())
    transformer_df = transformer_df.rename(columns={
                "sentiment": "transformer_sentiment",
                "sentiment_confidence": "transformer_confidence"
            })
    df = pd.concat([df, transformer_df], axis=1)
    df = predictor.predict(df)

    return df

def main():

    files = glob.glob("data/raw/*.csv")

    for path in files:

        print(f"Predicting: {path}")
        df = pd.read_csv(path)
        result = process_dataframe(df)
        output_path = "data/predictions/" + path.split("/")[-1].replace(".csv", "_predictions.parquet")
        result.to_parquet(output_path, index=False,)

        del result
        print(f"Saved: {output_path}")
    
if __name__ == "__main__": main()