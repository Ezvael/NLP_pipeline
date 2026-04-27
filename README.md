# Comments Analysis Pipeline

An end-to-end NLP pipeline for analysing **Russian marketplace reviews** (Ozon, Wildberries, AliExpress, etc.).

Given a CSV of raw comments the pipeline will:
1. **Preprocess** text — clean, lemmatize, apply regex-based pre-clustering.
2. **Label** comments with an LLM — assign a topic cluster and sentiment (optional, used to build training data).
3. **Train** lightweight ML models (Logistic Regression + Linear SVM with TF-IDF) for cluster and sentiment classification.
4. **Predict** cluster and sentiment on new, unlabelled data — combining ML models with a RuBERT transformer for sentiment.
5. **Build** a merged dashboard-ready dataset from multiple sources.

---

## Supported Labels

| Type | Labels |
|------|--------|
| **Cluster** | `chatbot` · `pricing` · `recommendations` · `suggest` · `delay` · `none` |
| **Sentiment** | `positive` · `neutral` · `negative` |

---

## Project Structure

```
comments-analysis/
├── main.py                    # CLI entry point — runs any pipeline mode
├── requirements.txt           # Python dependencies
├── config.example.json        # API key template (copy → config.json)
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # Text cleaning, lemmatization, regex clustering
│   ├── data_loader.py         # JSON → CSV conversion, dataset loading
│   ├── ai_labeling.py         # LLM-based cluster + sentiment labeling
│   ├── ml_models.py           # TF-IDF + LR/SVM training & evaluation
│   ├── sentiment_model.py     # RuBERT transformer sentiment inference
│   ├── predict.py             # Prediction pipeline (loads saved models)
│   └── dataset_builder.py     # Merge post metadata + comments for dashboard
│
└── models/                    # Saved .pkl model artefacts (gitignored)
```

---

## Installation

Requires **Python 3.9+**.

```bash
git clone https://github.com/<your-username>/comments-analysis.git
cd comments-analysis
pip install -r requirements.txt
```

Download the required NLTK data once:

```python
import nltk
nltk.download("stopwords")
```

The RuBERT transformer model (`blanchefort/rubert-base-cased-sentiment`) is downloaded automatically by `transformers` on first use.

---

## Configuration

Some pipeline steps call an external LLM API. Copy the example config and fill in your credentials:

```bash
cp config.example.json config.json
```

Edit `config.json`:

```json
{
  "api_key":  "YOUR_API_KEY_HERE",
  "base_url": "https://litellm.tokengate.ru/v1",
  "model":    "deepseek/deepseek-chat"
}
```

Any OpenAI-compatible endpoint works (DeepSeek, OpenAI, Groq, LiteLLM proxy, etc.).

---

## Input Data Format

The pipeline expects a CSV file with at minimum one column:

| Column | Description |
|--------|-------------|
| `Комментарий` | Raw comment text (Russian) |

For **training** you also need:

| Column | Description |
|--------|-------------|
| `cluster_mode` | Ground-truth cluster label (from AI labeling or manual annotation) |
| `sentiment_mode` | Ground-truth sentiment label |

---

## Pipeline Walkthrough

### Step 0 — Load raw data (optional)

If your raw data is in JSON format (`url → [comment, comment, ...]`), convert it to CSV first:

```python
from src.data_loader import json_to_csv

json_to_csv(
    input_file="data/raw/ozon_comments.json",
    output_file="data/raw_datasets/ozon_comments.csv",
)
```

### Step 1 — Preprocess text

`src/preprocessing.py` does three things:

1. **Regex pre-clustering** — scans the raw text for keyword patterns and adds a `clusters` column (list of matched cluster names). This is a fast heuristic that catches obvious cases before any model is applied.
2. **Lemmatization** — tokenizes with `TweetTokenizer`, removes Russian stopwords, and lemmatizes each token with `pymorphy3`. The result is stored in a new `new comment` column.

```python
from src.preprocessing import preprocess_data
import pandas as pd

df = pd.read_csv("data/raw_datasets/ozon_comments.csv")
df_processed = preprocess_data(df, text_column="Комментарий")

# df_processed now has two extra columns:
#   clusters    — list of regex-matched cluster names (may be empty)
#   new comment — lemmatized, stopword-free text used by ML models
```

You can also call the underlying helpers independently:

```python
from src.preprocessing import lemmatize, classify_multi, patterns

# Lemmatize a single string
clean = lemmatize("Долго ждала заказ, всё пришло разбитым!")
# → "долго ждать заказ прийти разбить"

# Apply regex patterns to a DataFrame
df = classify_multi(df, text_col="Комментарий", patterns=patterns)
```

### Step 2 — Label with an LLM (build training data)

Use a large language model to assign cluster and sentiment labels. This step produces the ground-truth data needed for ML model training.

```bash
python main.py label \
    --input_file  data/raw_datasets/ozon_comments.csv \
    --output_file data/labeled/ozon_labeled.csv \
    --config      config.json \
    --batch_size  80
```

Or call the Python API directly:

```python
import json
from src.ai_labeling import label_dataset

with open("config.json") as f:
    cfg = json.load(f)

df_labeled = label_dataset(
    df,
    api_key=cfg["api_key"],
    base_url=cfg["base_url"],
    model=cfg["model"],
)
# Adds columns: predicted_cluster, predicted_sentiment
df_labeled.to_csv("data/labeled/ozon_labeled.csv", index=False)
```

After labeling, rename the prediction columns to the training-expected names:

```python
df_labeled = df_labeled.rename(columns={
    "predicted_cluster":  "cluster_mode",
    "predicted_sentiment": "sentiment_mode",
})
```

### Step 3 — Train ML models

Train Logistic Regression and Linear SVM with TF-IDF on the labelled data. The best model for each task is selected automatically and saved as `.pkl` files.

```bash
python main.py train \
    --input_file data/labeled/ozon_labeled.csv \
    --model_dir  models/
```

Models saved:

```
models/
├── cluster_vectorizer.pkl
├── sentiment_vectorizer.pkl
├── cluster_label_encoder.pkl
├── sentiment_label_encoder.pkl
├── best_cluster_model.pkl
└── best_sentiment_model.pkl
```

### Step 4 — Predict on new data

Run the full prediction pipeline on new, unlabelled comments.

```bash
python main.py predict \
    --input_file  data/raw_datasets/new_comments.csv \
    --output_file predictions.csv \
    --model_dir   models/
```

Add `--skip_transformer_sentiment` to skip RuBERT inference (much faster):

```bash
python main.py predict \
    --input_file  data/raw_datasets/new_comments.csv \
    --output_file predictions.csv \
    --skip_transformer_sentiment
```

Output columns added to the CSV:

| Column | Description |
|--------|-------------|
| `clusters` | Regex pre-clustering result (list) |
| `new comment` | Lemmatized text |
| `predicted_cluster` | ML model cluster prediction |
| `predicted_ml_sentiment` | ML model sentiment prediction |
| `transformer_sentiment` | RuBERT sentiment prediction |
| `transformer_sentiment_confidence` | RuBERT confidence score (0–1) |

### Step 5 — Build dashboard dataset (optional)

If you have JSON post-metadata files (mapping `url → {domain, date, ...}`) you can merge them with the labelled comments to produce a single flat table for a BI dashboard.

```bash
python main.py build_dashboard \
    --json_posts_folder  data/json_with_posts \
    --csv_posts_folder   data/csv_with_posts \
    --final_df_folder    data/final_df \
    --output_file        output_for_dash.csv
```

---

## Module Reference

### `src/preprocessing.py`

| Symbol | Description |
|--------|-------------|
| `preprocess_data(df, text_column)` | Full preprocessing: regex clustering + lemmatization |
| `lemmatize(text)` | Tokenize → remove stopwords → lemmatize (pymorphy3) |
| `classify_multi(df, text_col, patterns)` | Apply dict of regex patterns, store matched labels |
| `patterns` | Dict of compiled regex patterns for each cluster topic |

### `src/data_loader.py`

| Symbol | Description |
|--------|-------------|
| `json_to_csv(input_file, output_file)` | Convert raw JSON comments file → CSV |
| `json_posts_folder_to_csv(input_folder, output_folder)` | Batch-convert JSON post metadata → CSV |
| `combine_csv_files(folder_path)` | Stack all CSVs in a folder into one DataFrame |
| `load_dataset(filepath, text_column)` | Load CSV + validate required column exists |

### `src/ai_labeling.py`

| Symbol | Description |
|--------|-------------|
| `label_dataset(df, api_key, base_url, model, ...)` | Label full DataFrame via LLM batches |
| `classify_batch(client, batch, model)` | Single-batch LLM call, returns parsed JSON |

### `src/ml_models.py`

| Symbol | Description |
|--------|-------------|
| `train_and_evaluate_ml_models(df, text_col, cluster_col, sentiment_col)` | Train LR + SVM for both tasks, return artefacts dict |
| `evaluate_final_models(trained_models)` | Pick the best model per task on the test set |

### `src/predict.py`

| Symbol | Description |
|--------|-------------|
| `load_model_artifacts(model_dir)` | Load all `.pkl` files from a directory |
| `predict_on_new_data(df, artifacts, skip_transformer_sentiment)` | Run ML + transformer predictions |

### `src/sentiment_model.py`

| Symbol | Description |
|--------|-------------|
| `predict_transformer_sentiment(df, text_column, tags_column)` | RuBERT batch inference with optional tag boosting |
| `batch_sentiment(texts, batch_size)` | Raw probability output for each text |

### `src/dataset_builder.py`

| Symbol | Description |
|--------|-------------|
| `build_dashboard_dataset(json_posts_folder, csv_posts_folder, final_df_folder, output_path)` | End-to-end merge pipeline |
| `merge_metadata_with_comments(metadata_df, processed_df)` | Inner join on URL |

---

## How Preprocessing Was Chosen

Different preprocessing strategies were benchmarked on real data using Logistic Regression (F1 macro on cluster classification):

| Strategy | Valid F1 |
|----------|----------|
| Baseline (raw text) | lower |
| Stopword removal only | moderate |
| Stopwords + Stemming | moderate |
| **Stopwords + Lemmatization** | **highest** |
| Stopwords + Lemmatization + Bigrams | similar to above |

Lemmatization outperformed stemming for Russian text because morphological analysis (`pymorphy3`) correctly reduces inflected forms to their dictionary base form, preserving semantic information that stemming would truncate.

---

## How the Best ML Model Was Chosen

Logistic Regression and Linear SVM were evaluated with `RandomizedSearchCV` (F1 macro) across a grid of regularization parameters. The model with the higher accuracy on the held-out test set (30 % of the data) was kept. Both models consistently outperformed heavier alternatives due to the relatively small training set size and the high-dimensional but sparse TF-IDF feature space.

---

## Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you would like to change.
