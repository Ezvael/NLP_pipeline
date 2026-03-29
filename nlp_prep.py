import pandas as pd
import spacy
from joblib import Parallel, delayed
from stop_words import get_stop_words
import glob
from nlp_config import POS_SLANG, NEG_SLANG, PROFANITY, BRANDS, TOPIC_KEYWORDS
from nlp_features import (
    get_lemma, 
    normalize_elongation,
    extract_emoji_tags,
    detect_sarcasm,
    detect_negation,
    detect_topics,
    LAUGHTER_PATTERN,
    URL_PATTERN,
    EXCL_PATTERN,
    QUES_PATTERN,
    CYR_PATTERN
)

# initialisation
nlp = spacy.load("ru_core_news_sm", disable=["ner", "textcat"])
stopwords_ru = set(get_stop_words("ru"))

def preprocess_text(text):
    text = str(text).lower()
    tags = []

    tags += extract_emoji_tags(text)
    tags += detect_sarcasm(text)

    if EXCL_PATTERN.search(text):
        tags.append("TAG_EXCLAM_STRONG")
    if QUES_PATTERN.search(text):
        tags.append("TAG_QUESTION_STRONG")
    if LAUGHTER_PATTERN.search(text):
        tags.append("TAG_LAUGHTER")

    text = URL_PATTERN.sub(" ", text)
    text = CYR_PATTERN.sub(" ", text)

    doc = nlp(text)
    negated = detect_negation(doc)

    tokens = []

    for token in doc:
        word = token.text

        if word.startswith("#"):
            tags.append("TAG_HASHTAG")
            continue

        if word in BRANDS:
            tags.append(f"TAG_BRAND_{BRANDS[word]}")
            continue

        if word in stopwords_ru or len(word) < 3:
            continue

        word, elongated = normalize_elongation(word)
        if elongated:
            tags.append("TAG_ELONGATED")

        lemma = get_lemma(word)

        if token.i in negated:
            lemma = "NEG_" + lemma

        if lemma in POS_SLANG:
            tags.append("TAG_SLANG_POS")
        elif lemma in NEG_SLANG:
            tags.append("TAG_SLANG_NEG")

        if lemma in PROFANITY:
            tags.append("TAG_PROFANITY")

        tokens.append(lemma)

    topics = detect_topics(tokens)

    return {
        "processed_text": " ".join(tokens),
        "tags": tags,
        "topics": topics
    }

def process_file(input_path, output_path):
    df = pd.read_csv(input_path)

    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_convert(None)
    df["combined_text"] = df["title"].fillna("") + " " + df["post_text"].fillna("")

    results = Parallel(n_jobs=-1, backend="threading")(
        delayed(preprocess_text)(text)
        for text in df["combined_text"]
    )

    results_df = pd.DataFrame(results)
    df = pd.concat([df, results_df], axis=1)

    for topic in TOPIC_KEYWORDS:
        df[topic] = df["topics"].apply(lambda x: int(topic in x))

    df["tags_str"] = df["tags"].apply(lambda x: ",".join(x))
    df["topics_str"] = df["topics"].apply(lambda x: ",".join(x))

    df.to_parquet(output_path, index=False)

    print(f"Finished processing: {input_path}")

def run_preprocessing():
    input_files = glob.glob("data/pikabu_posts_*.csv")

    for input_path in input_files:
        name = input_path.split("pikabu_posts_")[-1].replace(".csv", "")
        output_path = f"data/preprocessed_pikabu_{name}.parquet"

        process_file(input_path, output_path)

    print("Preprocessing is completed!")

if __name__ == "__main__":
    run_preprocessing()