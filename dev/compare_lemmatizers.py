"""
compare_lemmatizers.py — Compare old vs new preprocessing for ML (LR/SVM).

Hypothesis: the old TweetTokenizer + NLTK + isalpha lemmatizer may produce
cleaner vocabulary for TF-IDF ML models, while the new LRU-cached lemmatizer
with len>=3 filter is better for RuBERT.

Experiment:
  1. Load a stratified 10k subsample from combined_labeled.csv (or the full
     balanced sample if the file is already small).
  2. Preprocess with BOTH lemmatizers.
  3. For each preprocessed variant train 2 models:
       - LogisticRegression  (max_iter=2000, class_weight='balanced')
       - LinearSVC            (max_iter=5000, class_weight='balanced')
     Both use a default TfidfVectorizer() — no tuning — so differences in
     score reflect the lemmatizer, not the TF-IDF parameters.
  4. Report F1 macro (cluster + sentiment) for each combination.

Run:
    python compare_lemmatizers.py
"""

import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

sys.path.insert(0, ".")

from src.preprocessing import lemmatize_new, lemmatize_old

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH   = Path(r"C:\Users\Deniz\PycharmProjects\pythonProject\Analyse comments"
                   r"\Datasets processing\combined_labeled.csv")
SAMPLE_SIZE = 10_000   # stratified sample per class
RANDOM_STATE = 42

CLUSTER_COL   = "cluster_mode"
SENTIMENT_COL = "sentiment_mode"
TEXT_COL      = "comment"        # canonical column name after alias normalisation

# ── Load & alias normalise ────────────────────────────────────────────────────

print("Loading data ...")
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

# Normalise text column
for alias in ("Комментарий", "Kommentarij"):
    if TEXT_COL not in df.columns and alias in df.columns:
        df = df.rename(columns={alias: TEXT_COL})
        break

for alias in ("predicted_cluster",):
    if CLUSTER_COL not in df.columns and alias in df.columns:
        df = df.rename(columns={alias: CLUSTER_COL})
for alias in ("predicted_sentiment",):
    if SENTIMENT_COL not in df.columns and alias in df.columns:
        df = df.rename(columns={alias: SENTIMENT_COL})

df = df.dropna(subset=[TEXT_COL, CLUSTER_COL, SENTIMENT_COL])
print(f"  {len(df):,} rows after dropna")

# ── Stratified sample ─────────────────────────────────────────────────────────

# Take up to SAMPLE_SIZE rows per cluster class to get a manageable but
# representative comparison dataset.
parts = []
for cls, grp in df.groupby(CLUSTER_COL):
    parts.append(grp.sample(min(SAMPLE_SIZE, len(grp)), random_state=RANDOM_STATE))
sample = pd.concat(parts, ignore_index=True)
print(f"  Stratified sample: {len(sample):,} rows")
print(f"  Cluster dist: {sample[CLUSTER_COL].value_counts().to_dict()}")
print(f"  Sentiment dist: {sample[SENTIMENT_COL].value_counts().to_dict()}")

# ── Lemmatize with both approaches ────────────────────────────────────────────

print("\nLemmatising with NEW lemmatizer (LRU + stop_words) ...")
t = time.time()
texts_new = sample[TEXT_COL].astype(str).apply(lemmatize_new)
print(f"  Done in {time.time()-t:.1f}s")

print("Lemmatising with OLD lemmatizer (TweetTokenizer + NLTK + isalpha) ...")
t = time.time()
texts_old = sample[TEXT_COL].astype(str).apply(lemmatize_old)
print(f"  Done in {time.time()-t:.1f}s")

# ── Helper: train & score ─────────────────────────────────────────────────────

def train_score(texts, targets, label=""):
    """Train LR + LinearSVC on texts, return (lr_f1, svm_f1)."""
    le = LabelEncoder()
    y = le.fit_transform(targets)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )

    vec = TfidfVectorizer()
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    lr  = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE)
    svm = LinearSVC(max_iter=5000, class_weight='balanced', random_state=RANDOM_STATE)

    lr.fit(Xtr, y_train)
    svm.fit(Xtr, y_train)

    lr_f1  = f1_score(y_test, lr.predict(Xte),  average='macro')
    svm_f1 = f1_score(y_test, svm.predict(Xte), average='macro')
    return lr_f1, svm_f1

# ── Run experiments ───────────────────────────────────────────────────────────

results = []

for lem_name, texts in [("new", texts_new), ("old", texts_old)]:
    for task_name, target_col in [("cluster", CLUSTER_COL), ("sentiment", SENTIMENT_COL)]:
        print(f"\nTraining [{lem_name} lem | {task_name}] ...")
        lr_f1, svm_f1 = train_score(texts, sample[target_col], label=f"{lem_name}/{task_name}")
        results.append({
            "lemmatizer": lem_name,
            "task":       task_name,
            "LR_f1_macro":  round(lr_f1,  4),
            "SVM_f1_macro": round(svm_f1, 4),
            "best_f1":      round(max(lr_f1, svm_f1), 4),
        })
        print(f"  LR  F1 macro = {lr_f1:.4f}")
        print(f"  SVM F1 macro = {svm_f1:.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
df_res = pd.DataFrame(results)
print(df_res.to_string(index=False))

print("\n-- Winner per task --")
for task in ["cluster", "sentiment"]:
    sub = df_res[df_res["task"] == task].copy()
    winner = sub.loc[sub["best_f1"].idxmax(), "lemmatizer"]
    delta  = sub["best_f1"].max() - sub["best_f1"].min()
    print(f"  {task:10s}: {winner} lemmatizer wins  (delta = {delta:+.4f})")

print("\nConclusion:")
print("  If 'old' wins cluster: use lemmatizer='old' in preprocess_data() for ML models,")
print("  keep lemmatizer='new' (default) for RuBERT transformer.")
print("  If 'new' wins or tie: keep single lemmatizer='new' for everything.")
print("=" * 60)
