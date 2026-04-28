import glob
import pandas as pd
from src.nlp.pipeline import NLPPipeline
from src.nlp.sentiment_analyzer import SentimentAnalyzer

CHUNK_SIZE = 5000
pipeline = NLPPipeline()
sentiment_model = SentimentAnalyzer(batch_size=16)

def process_chunk(df):

    results = []

    for text in df[
        "Комментарий"
    ].fillna(""):

        results.append(pipeline.process(text))

    features_df = pd.DataFrame(results)
    df = pd.concat([df.reset_index(drop=True), features_df], axis=1)
    sentiment_df = sentiment_model.predict(df["raw_text"].tolist())
    df = pd.concat([df, sentiment_df], axis=1)

    return df

def process_file(path):

    output_path = "data/processed/" + path.split("/")[-1].replace(".csv", ".parquet")
    first_chunk = True

    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):

        processed = process_chunk(chunk)
        processed.to_parquet(output_path, engine="pyarrow", append=not first_chunk, index=False)
        first_chunk = False

        del processed

    print(f"Finished: {path}")

def main():

    files = glob.glob("data/raw/*.csv")

    for path in files:

        process_file(path)

if __name__ == "__main__": main()