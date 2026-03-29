import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import glob
from tqdm import tqdm

# Model
MODEL_NAME = "blanchefort/rubert-base-cased-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Sets for semantics
BOOST_POS = {"TAG_EMO_POS", "TAG_SLANG_POS", "TAG_EXCLAM_STRONG"}
BOOST_NEG = {"TAG_EMO_NEG", "TAG_SLANG_NEG", "TAG_PROFANITY"}

# Sentiment
def batch_sentiment(texts, batch_size=32):
    all_probs = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Batch inference"):
        batch = texts[i:i+batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        probs = F.softmax(outputs.logits, dim=1).cpu().numpy()
        all_probs.extend(probs)

    return all_probs

# Process input files
def process_file(input_path):
    df = pd.read_parquet(input_path)

    # Extract brand from filename
    name = input_path.split("preprocessed_pikabu_")[-1].replace(".parquet", "")
    df["brand"] = name

    df["processed_text"] = df["processed_text"].fillna("")
    df["tags"] = df["tags"].apply(lambda x: x if isinstance(x, list) else [])

    probs = batch_sentiment(df["processed_text"].tolist())

    sentiments = []
    confidences = []

    tags_list = df["tags"].tolist()

    for i, (neg, neu, pos) in enumerate(probs):
        tags = tags_list[i]

        tag_pos = sum(t in BOOST_POS for t in tags)
        tag_neg = sum(t in BOOST_NEG for t in tags)

        alpha = 0.8
        beta = 0.2

        pos_final = alpha * pos + beta * (tag_pos * 0.1)
        neg_final = alpha * neg + beta * (tag_neg * 0.1)
        neu_final = alpha * neu

        total = pos_final + neg_final + neu_final
        pos_final /= total
        neg_final /= total
        neu_final /= total

        scores = {
            "positive": pos_final,
            "neutral": neu_final,
            "negative": neg_final
        }

        label = max(scores, key=scores.get)
        confidence = scores[label]

        sentiments.append(label)
        confidences.append(round(float(confidence), 4))

    df["sentiment"] = sentiments
    df["sentiment_confidence"] = confidences

    return df

# Main function
def run_semantic():
    input_files = glob.glob("data/preprocessed_pikabu_*.parquet")

    all_dfs = []

    for input_path in tqdm(input_files, desc="Processing files"):
        df_processed = process_file(input_path)
        all_dfs.append(df_processed)

    final_df = pd.concat(all_dfs, ignore_index=True)

    final_df.to_parquet("data/semantic_all_brands.parquet", index=False)

    print("Semantic part is completed!")

if __name__ == "__main__":
    run_semantic()