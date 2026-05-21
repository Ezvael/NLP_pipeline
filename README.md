# Comments Analysis Pipeline

An end-to-end NLP pipeline for analysing **Russian marketplace reviews** (Ozon, Wildberries, AliExpress, etc.).

Given raw comments the pipeline will:
1. **Convert** raw JSON comments to a flat CSV.
2. **Label** comments with an LLM — assign a topic cluster and sentiment (creates training data without manual annotation).
3. **Train** lightweight ML models (Logistic Regression + Linear SVM with TF-IDF, `class_weight='balanced'`) on the LLM-labeled data.
4. **Predict** cluster and sentiment on new, unlabelled data — combining ML models with a RuBERT transformer for sentiment.
5. **Build raw counts** — aggregate total comment volume per date and domain (dashboard denominator).
6. **Visualise** results in an interactive Streamlit dashboard.

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
│   ├── preprocessing.py         # Text cleaning, lemmatization, regex pre-filtering
│   ├── data_loader.py           # JSON → CSV conversion, date enrichment utilities
│   ├── ai_labeling.py           # LLM-based cluster + sentiment labeling
│   ├── ml_models.py             # TF-IDF + LR/SVM training & evaluation
│   ├── sentiment_model.py       # RuBERT transformer sentiment inference
│   └── predict.py               # Prediction pipeline (loads saved models)
│
├── dev/                         # Research & calibration utilities (not part of pipeline)
│   ├── compare_lemmatizers.py       # Compare old vs new lemmatizer on a sample
│   ├── compare_llm_labelers.py      # Compare multiple LLMs; train LR+SVM on each set of labels
│   ├── evaluate_prompt.py           # Evaluate prompt quality against a calibration file
│   └── make_calibration_sample.py   # Build a sample for manual annotation
│
├── models/                      # Saved .pkl model artefacts (gitignored)
│   ├── best_cluster_model.pkl
│   ├── best_sentiment_model.pkl
│   ├── cluster_vectorizer.pkl
│   ├── sentiment_vectorizer.pkl
│   ├── cluster_label_encoder.pkl
│   ├── sentiment_label_encoder.pkl
│   └── raw_counts_per_date.csv      # Total comments per date+domain (dashboard denominator)
│
├── examples/                    # Minimal real-format data samples
│   ├── json/
│   │   ├── comments_sample.json     # {url: [comment, ...]} — 3 posts, 9 comments
│   │   └── posts_sample.json        # {url: {domain, date, id_group, id_post}}
│   └── csv/
│       ├── labeled_sample.csv       # 10 rows: prediction output format
│       └── raw_counts_sample.csv    # 10 rows: date, domain, total_comments
│
└── logs/                        # Runtime logs (gitignored)
```

---

## Input Data Formats

### JSON comments file (`convert` / `build_raw_counts`)

```json
{
  "https://vk.com/wall-123_456": ["Comment one.", "Comment two."],
  "https://vk.com/wall-123_457": ["Another comment."]
}
```

See `examples/json/comments_sample.json` for a concrete example.

### JSON posts metadata file (`build_raw_counts` / date enrichment)

```json
{
  "https://vk.com/wall-123_456": {
    "domain": "ozon",
    "date": 1700000000,
    "id_group": 123,
    "id_post": 456
  }
}
```

`date` is a Unix timestamp (integer). `domain` is a free-form platform identifier.
See `examples/json/posts_sample.json` for a concrete example.

### Analysis-ready CSV (for `train` / `predict` / dashboard)

| Column | Required | Description |
|--------|----------|-------------|
| `comment` | yes | Comment text. Aliases: `Комментарий`, `Kommentarij` |
| `url` | no | Post URL — used to join date and domain from post metadata |
| `cluster_mode` | for train | Cluster label. Aliases: `predicted_cluster`, `cluster` |
| `sentiment_mode` | for train | Sentiment label. Aliases: `predicted_sentiment`, `sentiment` |
| `date` | no | Post date — enables time-series charts in dashboard |
| `domain` | no | Platform name — enables domain filter in dashboard |

See `examples/csv/labeled_sample.csv` for a concrete example.

### Raw counts CSV (`models/raw_counts_per_date.csv`)

| Column | Description |
|--------|-------------|
| `date` | ISO date string |
| `domain` | Platform name |
| `total_comments` | Total comments collected that day for that domain |

Used as the denominator for the "Cluster mentions vs. total volume" chart.
See `examples/csv/raw_counts_sample.csv` for a concrete example.

---

## Quick Start

```
data/
  ozon_comments.json   # {url: [comment, ...]}
models/
  *.pkl                # pre-trained model artefacts
  raw_counts_per_date.csv
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

## CLI Reference

```
python main.py <mode> [options]
```

### `convert` — JSON → flat CSV

Accepts either a **single file** or an **entire directory** of JSON files.
In both cases the filename stem (without `.json`) is written to a `domain` column.

```bash
# Single file — domain = "ozon_comments" (stem of the filename)
python main.py convert \
    --input_file  data/raw/ozon_comments.json \
    --output_file data/flat.csv

# Directory — every .json file becomes a separate domain
python main.py convert \
    --input_dir   "path/to/Raw json comments/" \
    --output_file data/flat.csv
```

Input format: `{ "url": ["comment 1", "comment 2"], ... }`  
Output columns: `url`, `comment`, `domain`.

### `label` — Label raw data with an LLM

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

### `build_raw_counts` — Rebuild total comment volume per date+domain

```bash
python main.py build_raw_counts \
    --posts_dir    "path/to/Json with posts" \
    --comments_dir "path/to/Raw json comments" \
    --output_file  models/raw_counts_per_date.csv
```

Reads all `.json` files from `--posts_dir` (url → domain+date) and `--comments_dir`
(url → list of comments), joins them, and aggregates total comment count by date and domain.

### `dashboard` — Launch Streamlit dashboard

```bash
python main.py dashboard --input_file predictions.csv
# or directly:
python -m streamlit run dashboard.py -- --input_file predictions.csv
```

By default the dashboard looks for `models/raw_counts_per_date.csv` (already present in the
repository). To point it at a different file use `--raw_counts`:

```bash
python main.py dashboard \
    --input_file predictions.csv \
    --raw_counts path/to/other_raw_counts.csv
```

---

## Dashboard

`dashboard.py` visualises a predictions CSV interactively.

**Sidebar filters:** sentiment · domain · cluster · date range · free-text search

**KPIs:** total posts · avg comment length · positive % · negative % · toxic %

**Charts:**
- Sentiment distribution (bar)
- Sentiment over time (line)
- Cluster distribution (bar)
- Cluster mentions vs. total volume — grouped bars per period + total comments on secondary axis (week / month / quarter). The domain filter also narrows the volume denominator.
- Sarcasm usage (pie) — shown when a `tags` column is present

**Posts table:** up to 200 filtered rows.

### Raw counts file

The "Cluster mentions vs. total volume" chart requires `models/raw_counts_per_date.csv`.
This file is included in the repository (pre-built from the training dataset).

To rebuild it from your own data:

```bash
python main.py build_raw_counts \
    --posts_dir    "path/to/Json with posts" \
    --comments_dir "path/to/Raw json comments" \
    --output_file  models/raw_counts_per_date.csv
```

It reads all `.json` files from `--posts_dir` (url → domain + Unix timestamp) and
`--comments_dir` (url → list of comments), joins them by URL, and aggregates the total
comment count by date and domain. The result is used as the denominator in the volume chart.

---

## Model Performance

Evaluated on a 30% held-out test split (random_state=0) of the combined VK dataset.

### Cluster — LinearSVC (macro F1 = 0.578)

| Class | Precision | Recall | F1 | Random F1 | Support |
|-------|-----------|--------|----|-----------|---------|
| chatbot | 0.648 | 0.699 | 0.673 | 0.033 | 871 |
| delay | 0.765 | 0.756 | 0.760 | 0.451 | 15 174 |
| no_cluster | 0.760 | 0.763 | 0.761 | 0.500 | 16 730 |
| pricing | 0.519 | 0.550 | 0.534 | 0.028 | 1 018 |
| recommendations | 0.375 | 0.103 | 0.162 | 0.000 | 29 |
| **macro avg** | **0.613** | **0.574** | **0.578** | **0.202** | — |

### Sentiment — LinearSVC (macro F1 = 0.695)

| Class | Precision | Recall | F1 | Random F1 | Support |
|-------|-----------|--------|----|-----------|---------|
| negative | 0.789 | 0.862 | 0.824 | 0.472 | 16 059 |
| neutral | 0.847 | 0.778 | 0.811 | 0.499 | 16 857 |
| positive | 0.481 | 0.425 | 0.451 | 0.036 | 906 |
| **macro avg** | **0.706** | **0.688** | **0.695** | **0.336** | — |

---

## dev/ Tools Reference

These scripts are **not part of the prediction pipeline** — they support research, calibration, and lemmatizer comparison.

### `compare_lemmatizers.py`
Compares the old lemmatizer (TweetTokenizer + NLTK stopwords) with the new one (pymorphy3 + stop_words library + LRU cache) on a random sample.

```bash
python dev/compare_lemmatizers.py
```

### `compare_llm_labelers.py`
Labels a sample with one or more LLM models, trains LR + LinearSVC on each set of labels, and compares macro F1. Supports resumable overnight runs via checkpoints.

```bash
python dev/compare_llm_labelers.py \
    --input_file  "path/to/ozon_comments.json" \
    --config      config.json \
    --models      "deepseek/deepseek-v4-pro" "openai/gpt-4o-mini" \
    --output_file dev/comparison_results.csv
```

### `evaluate_prompt.py`
Runs the current prompt on a manually annotated calibration file and reports per-class precision/recall/F1 for cluster and sentiment, plus a confusion matrix and top-15 errors.

```bash
python dev/evaluate_prompt.py \
    --calibration_file dev/calibration_sample.xlsx \
    --model            deepseek/deepseek-v4-pro
```

### `make_calibration_sample.py`
Builds a CSV for manual annotation — stratified across cluster keyword filters to ensure informative coverage.

```bash
python dev/make_calibration_sample.py \
    --input_file  "path/to/ozon_comments.json" \
    --output_file dev/calibration_sample.csv \
    --n_per_class 20 \
    --n_none      30
```

---

## Module Reference

### `src/data_loader.py`
| Symbol | Description |
|--------|-------------|
| `json_to_csv(input_file, output_file, domain=None)` | Convert a single JSON file → flat CSV (url, comment[, domain]) |
| `json_dir_to_csv(input_dir, output_file)` | Convert a directory of JSON files → flat CSV (url, comment, domain) |
| `enrich_with_post_dates(df, posts_dir)` | Join unix dates from JSON post files onto df by url |
| `combine_csv_files(folder_path)` | Stack all CSVs in a folder into one DataFrame |
| `load_dataset(filepath, text_column)` | Load CSV + validate required column exists |

### `src/preprocessing.py`
| Symbol | Description |
|--------|-------------|
| `preprocess_data(df, text_column)` | Clean → lemmatize → regex cluster label |
| `lemmatize_new(text)` | LRU-cached pymorphy3 lemmatization (default) |
| `lemmatize_old(text)` | TweetTokenizer + NLTK lemmatization (legacy) |
| `patterns` | Dict of compiled regex patterns for cluster pre-filtering |

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
