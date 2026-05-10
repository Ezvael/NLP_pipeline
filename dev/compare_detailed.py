"""Per-class classification reports for both lemmatizers."""
import sys
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

sys.path.insert(0, ".")
from src.preprocessing import lemmatize_new, lemmatize_old

DATA_PATH    = Path(r"C:\Users\Deniz\PycharmProjects\pythonProject\Analyse comments"
                    r"\Datasets processing\combined_labeled.csv")
RANDOM_STATE = 42

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
for alias in ("Комментарий", "Kommentarij"):
    if "comment" not in df.columns and alias in df.columns:
        df = df.rename(columns={alias: "comment"})
        break
for src, dst in [("predicted_cluster", "cluster_mode"), ("predicted_sentiment", "sentiment_mode")]:
    if dst not in df.columns and src in df.columns:
        df = df.rename(columns={src: dst})

df = df.dropna(subset=["comment", "cluster_mode", "sentiment_mode"])

parts = []
for cls, grp in df.groupby("cluster_mode"):
    parts.append(grp.sample(min(10_000, len(grp)), random_state=RANDOM_STATE))
sample = pd.concat(parts, ignore_index=True)
print(f"Sample: {len(sample):,} rows")

print("Lemmatising NEW ...")
texts_new = sample["comment"].astype(str).apply(lemmatize_new)
print("Lemmatising OLD ...")
texts_old = sample["comment"].astype(str).apply(lemmatize_old)


def best_report(texts, targets):
    le = LabelEncoder()
    y  = le.fit_transform(targets)
    Xtr_raw, Xte_raw, y_train, y_test = train_test_split(
        texts, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )
    vec = TfidfVectorizer()
    Xtr = vec.fit_transform(Xtr_raw)
    Xte = vec.transform(Xte_raw)

    lr  = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    svm = LinearSVC(max_iter=5000,          class_weight="balanced", random_state=RANDOM_STATE)
    lr.fit(Xtr, y_train)
    svm.fit(Xtr, y_train)

    lr_f1  = f1_score(y_test, lr.predict(Xte),  average="macro")
    svm_f1 = f1_score(y_test, svm.predict(Xte), average="macro")

    if svm_f1 >= lr_f1:
        preds, model_name = svm.predict(Xte), "LinearSVC"
    else:
        preds, model_name = lr.predict(Xte),  "LogisticRegression"

    report = classification_report(
        y_test, preds, target_names=le.classes_, digits=3
    )
    return report, model_name


SEP = "=" * 58
for lem_name, texts in [("NEW (LRU + stop_words)", texts_new),
                         ("OLD (TweetTokenizer + NLTK)", texts_old)]:
    for task, col in [("CLUSTER", "cluster_mode"), ("SENTIMENT", "sentiment_mode")]:
        report, model_name = best_report(texts, sample[col])
        print(f"\n{SEP}")
        print(f"  {lem_name} | {task} | best={model_name}")
        print(SEP)
        print(report)
