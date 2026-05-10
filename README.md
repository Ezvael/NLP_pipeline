# Comments Analysis Pipeline

An end-to-end NLP pipeline for analysing **Russian marketplace reviews** (Ozon, Wildberries, AliExpress, etc.).

Given raw comments the pipeline will:
1. **Convert** raw JSON comments to a flat CSV.
2. **Label** comments with an LLM — assign a topic cluster and sentiment (used to build training data).
3. **Train** lightweight ML models (Logistic Regression + Linear SVM with TF-IDF, `class_weight='balanced'`) for cluster and sentiment classification.
4. **Predict** cluster and sentiment on new, unlabelled data — combining ML models with a RuBERT transformer for sentiment.
5. **Visualise** results in an interactive Streamlit dashboard.

---

## Supported Labels

| Type | Labels |
|------|--------|
| **Cluster** | `chatbot` · `pricing` · `recommendations` · `delay` · `none` |
| **Sentiment** | `positive` · `neutral` · `negative` |

---

## Project Structure

```
full_prediction_project/
├── main.py                      # CLI entry point — runs any pipeline mode
├── dashboard.py                 # Streamlit analytics dashboard
├── requirements.txt
├── config.example.json          # API key template (copy → config.json)
├── .gitignore
│
├── src/
│   ├── config.py                # Shared constants
│   ├── cleaner.py               # URL removal, lowercase, whitespace normalisation
│   ├── feature_extractor.py     # Emoji, slang and profanity tag detection
│   ├── topic_detector.py        # Token-based keyword topic detection
│   ├── preprocessing.py         # Text cleaning, lemmatization, regex clustering
│   ├── data_loader.py           # JSON → CSV conversion, dataset loading utilities
│   ├── dataset_builder.py       # Merge post metadata + comments for dashboard
│   ├── ai_labeling.py           # LLM-based cluster + sentiment labeling
│   ├── ml_models.py             # TF-IDF + LR/SVM training & evaluation
│   ├── sentiment_model.py       # RuBERT transformer sentiment inference
│   └── predict.py               # Prediction pipeline (loads saved models)
│
├── dev/                         # Development & research utilities (not part of pipeline)
│   ├── compare_lemmatizers.py
│   └── compare_detailed.py
│
├── models/                      # Saved .pkl model artefacts (gitignored)
│   ├── best_cluster_model.pkl
│   ├── best_sentiment_model.pkl
│   ├── cluster_vectorizer.pkl
│   ├── sentiment_vectorizer.pkl
│   ├── cluster_label_encoder.pkl
│   ├── sentiment_label_encoder.pkl
│   ├── raw_counts_per_date.csv      # Total comments per date (dashboard denominator)
│   └── training_stats_balanced.txt
│
└── logs/                        # Runtime logs (gitignored)
```

---

## Ozon Pipeline — Quick Start

```
data/
  ozon_comments.json   # {url: [comment, ...]}
  ozon_posts.csv       # url, date, domain  (one row per post)
models/
  *.pkl                # pre-trained models
```

```bash
# 1. JSON → flat CSV (url + comment, one row per comment)
python main.py convert \
    --input_file  data/ozon_comments.json \
    --output_file data/ozon_flat.csv

# 2. Predict cluster + sentiment
python main.py predict \
    --input_file  data/ozon_flat.csv \
    --output_file predictions.csv

# 3. Launch dashboard
python main.py dashboard --input_file predictions.csv
```

---

## Installation

Requires **Python 3.9+**.

```bash
git clone https://github.com/<your-username>/comments-analysis.git
cd comments-analysis
pip install -r requirements.txt
```

The RuBERT transformer model (`blanchefort/rubert-base-cased-sentiment`) is downloaded
automatically by `transformers` on first use.

---

## Configuration

```bash
cp config.example.json config.json
```

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

| Column | Description |
|--------|-------------|
| `Комментарий` | Raw comment text (Russian). Aliases: `Kommentarij`, `comment` |

For **training** you also need:

| Column | Description |
|--------|-------------|
| `cluster_mode` | Ground-truth cluster label (aliases: `predicted_cluster`, `cluster`) |
| `sentiment_mode` | Ground-truth sentiment label (aliases: `predicted_sentiment`, `sentiment`) |

---

## CLI Reference (`main.py`)

```
python main.py <mode> [options]
```

### `convert` — JSON → flat CSV

```bash
python main.py convert \
    --input_file  data/ozon_comments.json \
    --output_file data/ozon_flat.csv
```

Input JSON format: `{ "https://url/...": ["comment 1", "comment 2"], ... }`  
Output CSV columns: `url`, `comment`.

### `label` — AI-label raw data

```bash
python main.py label \
    --input_file  data/ozon_flat.csv \
    --output_file data/labeled/ozon_labeled.csv \
    --config      config.json \
    --batch_size  80
```

### `train` — Train ML models

```bash
python main.py train \
    --input_file data/labeled/ozon_labeled.csv \
    --model_dir  models/
```

### `predict` — Run predictions on new data

```bash
python main.py predict \
    --input_file  data/ozon_flat.csv \
    --output_file predictions.csv \
    --model_dir   models/
```

Add `--skip_transformer_sentiment` to skip RuBERT inference (faster, ML sentiment only).

### `build_dashboard` — Merge post metadata

```bash
python main.py build_dashboard \
    --json_posts_folder  data/json_with_posts \
    --csv_posts_folder   data/csv_with_posts \
    --final_df_folder    data/final_df \
    --output_file        output_for_dash.csv
```

### `dashboard` — Launch Streamlit dashboard

```bash
python main.py dashboard --input_file predictions.csv
```

Or directly:

```bash
python -m streamlit run dashboard.py -- --input_file predictions.csv
```

---

## Dashboard

`dashboard.py` visualises `predictions.csv` interactively.

**Sidebar filters:** sentiment · domain · cluster · date range · full-text search

**KPIs:** total posts · avg length · positive % · negative % · toxic %

**Charts:**
- Sentiment distribution (bar)
- Sentiment over time (line)
- Cluster distribution (bar)
- Cluster mentions vs. total volume — grouped bars per period + total comments on secondary axis (week / month / quarter)
- Sarcasm usage (pie) — shown when a `tags` column is present

**Posts table:** up to 200 filtered rows.

The "Cluster mentions vs. total volume" chart requires `models/raw_counts_per_date.csv`
(generated automatically by `predict` mode when `--posts_csv` is provided, or separately
via `make_ozon_date_counts.py`).

---

## Output Columns

After `predict`, `predictions.csv` contains:

| Column | Source | Description |
|--------|--------|-------------|
| `comment` | input | Raw comment text |
| `new comment` | preprocessing | Lemmatized, stopword-free text |
| `clusters` | regex | Regex-matched cluster heuristics |
| `predicted_cluster` | ML | LinearSVC cluster prediction |
| `predicted_ml_sentiment` | ML | LinearSVC sentiment prediction |
| `transformer_sentiment` | RuBERT | `positive` / `neutral` / `negative` |
| `transformer_sentiment_confidence` | RuBERT | Confidence score 0–1 |

---

## Model Performance

Training set: ~104k rows (all minority-cluster rows + 60k sampled `none`).

### Cluster — LinearSVC

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| chatbot | 0.57 | 0.56 | 0.57 |
| delay | 0.70 | 0.73 | 0.71 |
| none | 0.92 | 0.90 | 0.91 |
| pricing | 0.37 | 0.43 | 0.40 |
| recommendations | 0.42 | 0.34 | 0.38 |
| **macro avg** | **0.59** | **0.57** | **0.58** |

### Sentiment — LogisticRegression

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| negative | 0.65 | 0.82 | 0.72 |
| neutral | 0.92 | 0.83 | 0.87 |
| positive | 0.08 | 0.08 | 0.08 |
| **macro avg** | **0.55** | **0.57** | **0.56** |

`positive` F1 is low due to extreme class imbalance (~0.4% of rows). RuBERT handles positive sentiment significantly better in practice.

---

## Module Reference

### `src/data_loader.py`
| Symbol | Description |
|--------|-------------|
| `json_to_csv(input_file, output_file)` | Convert raw JSON comments → flat CSV (url, comment) |
| `json_posts_folder_to_csv(input_folder, output_folder)` | Batch-convert JSON post metadata → CSV |
| `combine_csv_files(folder_path)` | Stack all CSVs in a folder into one DataFrame |
| `load_dataset(filepath, text_column)` | Load CSV + validate required column exists |

### `src/preprocessing.py`
| Symbol | Description |
|--------|-------------|
| `preprocess_data(df, text_column)` | Clean → lemmatize → regex cluster |
| `lemmatize(text)` | LRU-cached pymorphy3 lemmatization |

### `src/ml_models.py`
| Symbol | Description |
|--------|-------------|
| `train_and_evaluate_ml_models(df, ...)` | TF-IDF + RandomizedSearchCV over LR and LinearSVC |
| `evaluate_final_models(trained_models)` | Pick best model per task on held-out test set |

### `src/predict.py`
| Symbol | Description |
|--------|-------------|
| `load_model_artifacts(model_dir)` | Load all `.pkl` files from a directory |
| `predict_on_new_data(df, artifacts, ...)` | Run ML + optional transformer predictions |

### `src/sentiment_model.py`
| Symbol | Description |
|--------|-------------|
| `predict_transformer_sentiment(df, ...)` | RuBERT batch inference; auto-detects GPU |

### `src/dataset_builder.py`
| Symbol | Description |
|--------|-------------|
| `build_dashboard_dataset(...)` | End-to-end merge: JSON metadata → CSV → join with comments |
| `merge_metadata_with_comments(metadata_df, processed_df)` | Inner join on URL |
