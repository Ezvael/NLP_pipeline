import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configuration
MODEL_NAME = "blanchefort/rubert-base-cased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

# Sets for semantics boosting
BOOST_POS = {"TAG_EMO_POS", "TAG_SLANG_POS", "TAG_EXCLAM_STRONG"}
BOOST_NEG = {"TAG_EMO_NEG", "TAG_SLANG_NEG", "TAG_PROFANITY"}

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

def predict_transformer_sentiment(df, text_column='Комментарий', tags_column='tags'):
    df_copy = df.copy()
    df_copy[text_column] = df_copy[text_column].fillna("")
    if tags_column in df_copy.columns:
        df_copy[tags_column] = df_copy[tags_column].apply(lambda x: x if isinstance(x, list) else [])
    else:
        df_copy[tags_column] = [[] for _ in range(len(df_copy))]

    probs = batch_sentiment(df_copy[text_column].tolist())

    sentiments = []
    confidences = []
    tags_list = df_copy[tags_column].tolist()

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

    df_copy["transformer_sentiment"] = sentiments
    df_copy["transformer_sentiment_confidence"] = confidences
    
    return df_copy
