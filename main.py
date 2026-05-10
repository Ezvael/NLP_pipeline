"""
main.py — Entry point for the Comments Analysis pipeline.

The pipeline has five modes selected via --mode:

  label            AI-label a raw CSV with cluster + sentiment using an LLM.
                   Requires config.json with api_key, base_url, model.
  train            Preprocess data + train ML models + save artefacts.
                   Input CSV must have columns: comment, cluster_mode, sentiment_mode.
  predict          Preprocess data + load saved models + run predictions.
  build_dashboard  Merge post metadata (JSON) with labelled comments into one CSV.

Run any mode with --help for full argument list:
    python main.py train --help
    python main.py predict --help

Quick examples
--------------
# 1. Label raw data with an LLM (creates training ground-truth)
python main.py label \\
    --input_file data/raw_datasets/ozon_comments.csv \\
    --output_file data/labeled/ozon_labeled.csv

# 2. Train ML models on labelled data
python main.py train \\
    --input_file data/labeled/ozon_labeled.csv \\
    --output_file predictions.csv

# 3. Predict on new, unlabelled data
python main.py predict \\
    --input_file data/raw_datasets/new_comments.csv \\
    --output_file predictions.csv

# 4. Build dashboard-ready dataset
python main.py build_dashboard \\
    --json_posts_folder  data/json_with_posts \\
    --csv_posts_folder   data/csv_with_posts \\
    --final_df_folder    data/final_df \\
    --output_file        output_for_dash.csv
"""

import argparse
import json
import os
import pickle
import sys

import pandas as pd

from src.preprocessing import preprocess_data
from src.ml_models import train_and_evaluate_ml_models, evaluate_final_models
from src.predict import predict_on_new_data, load_model_artifacts

# Column aliases accepted as input (keys → canonical pipeline names)
_COLUMN_ALIASES = {
    "Комментарий":           "comment",        # old Cyrillic column name -> canonical
    "Kommentarij":           "comment",        # combined_labeled.csv ASCII alias
    "new_comment":           "new comment",
    "cluster":               "cluster_mode",
    "predicted_cluster":     "cluster_mode",   # LLM-labeled output used as train target
    "sentiment":             "sentiment_mode",
    "predicted_sentiment":   "sentiment_mode", # LLM-labeled output used as train target
}

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known column aliases to the canonical names the pipeline uses."""
    rename = {src: dst for src, dst in _COLUMN_ALIASES.items()
              if src in df.columns and dst not in df.columns}
    if rename:
        print(f"Renaming columns: {rename}")
    return df.rename(columns=rename)


# ---------------------------------------------------------------------------
# Mode: label
# ---------------------------------------------------------------------------

def run_label(args):
    """Label raw comments with cluster + sentiment using an LLM."""
    from src.ai_labeling import label_dataset

    if not os.path.exists(args.config):
        print(f"Error: config file '{args.config}' not found. "
              "Copy config.example.json → config.json and fill in your API key.")
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    print(f"Loading data from '{args.input_file}'...")
    df = pd.read_csv(args.input_file, encoding='utf-8-sig')

    df = _normalize_columns(df)
    if "comment" not in df.columns:
        print("Error: input CSV must contain a 'comment' (or 'Комментарий') column.")
        sys.exit(1)

    print(f"Labeling {len(df)} comments with model '{cfg['model']}'...")
    df_labeled = label_dataset(
        df,
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        model=cfg["model"],
        text_column="comment",
        batch_size=args.batch_size,
    )

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    df_labeled.to_csv(args.output_file, index=False, encoding='utf-8-sig')
    print(f"Labeled dataset saved to '{args.output_file}'")


# ---------------------------------------------------------------------------
# Mode: train
# ---------------------------------------------------------------------------

def run_train(args):
    """Preprocess + train ML models + save artefacts."""
    print(f"Loading data from '{args.input_file}'...")
    df = _normalize_columns(pd.read_csv(args.input_file, encoding='utf-8-sig'))

    if "comment" not in df.columns:
        print("Error: input CSV must contain a 'comment' (or 'Комментарий') column.")
        sys.exit(1)

    required_targets = ["cluster_mode", "sentiment_mode"]
    missing = [c for c in required_targets if c not in df.columns]
    if missing:
        print(f"Error: training requires columns {required_targets}. Missing: {missing}")
        print("Tip: run 'label' mode first, then rename predicted_cluster → cluster_mode "
              "and predicted_sentiment → sentiment_mode.")
        sys.exit(1)

    if "new comment" in df.columns:
        print("Preprocessed text column found — skipping preprocessing step.")
        df_processed = df.copy()
    else:
        print("Preprocessing text...")
        df_processed = preprocess_data(df.copy())

    df_trainable = df_processed.dropna(subset=["cluster_mode", "sentiment_mode"]).copy()
    if df_trainable.empty:
        print("Error: no valid rows after dropping NaNs in target columns.")
        sys.exit(1)

    print(f"Training on {len(df_trainable)} rows...")
    trained_artifacts = train_and_evaluate_ml_models(
        df_trainable, "new comment", "cluster_mode", "sentiment_mode"
    )
    final_models = evaluate_final_models(trained_artifacts)
    trained_artifacts.update(final_models)

    # Save artefacts
    os.makedirs(args.model_dir, exist_ok=True)
    skip_keys = {"X_cluster_test", "c_test", "X_sentiment_test", "s_test"}
    for key, value in trained_artifacts.items():
        if key not in skip_keys:
            with open(os.path.join(args.model_dir, f"{key}.pkl"), "wb") as f:
                pickle.dump(value, f)
    print(f"Models saved to '{args.model_dir}/'")


# ---------------------------------------------------------------------------
# Mode: predict
# ---------------------------------------------------------------------------

def run_predict(args):
    """Preprocess + load saved models + run predictions."""
    print(f"Loading data from '{args.input_file}'...")
    df = _normalize_columns(pd.read_csv(args.input_file, encoding='utf-8-sig'))

    if "comment" not in df.columns:
        print("Error: input CSV must contain a 'comment' (or 'Комментарий') column.")
        sys.exit(1)

    if "new comment" in df.columns:
        print("Preprocessed text column found — skipping preprocessing step.")
        df_processed = df.copy()
    else:
        print("Preprocessing text...")
        df_processed = preprocess_data(df.copy())

    print(f"Loading model artefacts from '{args.model_dir}/'...")
    try:
        artifacts = load_model_artifacts(args.model_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run 'python main.py train ...' first to create the model files.")
        sys.exit(1)

    print("Running predictions...")
    df_result = predict_on_new_data(
        df_processed,
        artifacts,
        skip_transformer_sentiment=args.skip_transformer_sentiment,
    )

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    df_result.to_csv(args.output_file, index=False, encoding='utf-8-sig')
    print(f"Predictions saved to '{args.output_file}'")

    if args.posts_csv:
        posts = pd.read_csv(args.posts_csv, usecols=["url", "date", "domain"])

        # Merge date + domain into predictions
        df_result = pd.read_csv(args.output_file, encoding="utf-8-sig")
        df_result = df_result.merge(posts[["url", "date", "domain"]], on="url", how="left")
        df_result.to_csv(args.output_file, index=False, encoding="utf-8-sig")
        n_dated = df_result["date"].notna().sum()
        print(f"  Enriched with dates/domain from '{args.posts_csv}' ({n_dated:,} rows matched)")

        # Save raw_counts_per_date.csv for dashboard denominator
        raw_counts_path = os.path.join(args.model_dir, "raw_counts_per_date.csv")
        date_col = pd.to_datetime(posts["date"], unit="s", errors="coerce")
        rc = (date_col.dt.date
              .value_counts()
              .rename_axis("date")
              .reset_index(name="total_comments")
              .sort_values("date"))
        rc.to_csv(raw_counts_path, index=False)
        print(f"  Raw counts per date saved → {raw_counts_path} ({len(rc)} unique dates)")


# ---------------------------------------------------------------------------
# Mode: convert
# ---------------------------------------------------------------------------

def run_convert(args):
    """Convert a raw JSON comments file to a flat CSV (url, comment)."""
    from src.data_loader import json_to_csv

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    json_to_csv(args.input_file, args.output_file)


# ---------------------------------------------------------------------------
# Mode: dashboard
# ---------------------------------------------------------------------------

def run_dashboard(args):
    """Launch the Streamlit analytics dashboard."""
    import subprocess
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "dashboard.py",
         "--", "--input_file", args.input_file],
        check=True,
    )


# ---------------------------------------------------------------------------
# Mode: build_dashboard
# ---------------------------------------------------------------------------

def run_build_dashboard(args):
    """Merge post metadata with labelled comments into a dashboard-ready CSV."""
    from src.dataset_builder import build_dashboard_dataset

    build_dashboard_dataset(
        json_posts_folder=args.json_posts_folder,
        csv_posts_folder=args.csv_posts_folder,
        final_df_folder=args.final_df_folder,
        output_path=args.output_file,
    )


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Comments Analysis Pipeline — cluster and sentiment prediction for marketplace reviews.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # ── convert ────────────────────────────────────────────────────────────
    p_conv = sub.add_parser("convert",
                            help="Convert a raw JSON comments file to a flat CSV (url, comment).")
    p_conv.add_argument("--input_file",  required=True,
                        help="Path to source .json file ({url: [comment, ...]} format).")
    p_conv.add_argument("--output_file", default="data/ozon_flat.csv",
                        help="Where to save the flat CSV (default: data/ozon_flat.csv).")

    # ── label ──────────────────────────────────────────────────────────────
    p_label = sub.add_parser("label", help="Label a raw CSV with cluster + sentiment via LLM.")
    p_label.add_argument("--input_file", required=True,
                         help="Path to input CSV (must have 'comment' column).")
    p_label.add_argument("--output_file", default="data/labeled/labeled.csv",
                         help="Where to save the labeled CSV.")
    p_label.add_argument("--config", default="config.json",
                         help="Path to config.json with api_key, base_url, model.")
    p_label.add_argument("--batch_size", type=int, default=80,
                         help="Number of comments per LLM API call (default: 80).")

    # ── train ──────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Train ML models on labelled data.")
    p_train.add_argument("--input_file", required=True,
                         help="Labeled CSV with 'comment', 'cluster_mode', 'sentiment_mode'.")
    p_train.add_argument("--model_dir", default="models",
                         help="Directory to save trained model .pkl files (default: models/).")

    # ── predict ────────────────────────────────────────────────────────────
    p_pred = sub.add_parser("predict", help="Run cluster + sentiment prediction on new data.")
    p_pred.add_argument("--input_file", required=True,
                        help="Path to input CSV (must have 'comment' column).")
    p_pred.add_argument("--output_file", default="predictions.csv",
                        help="Where to save predictions CSV (default: predictions.csv).")
    p_pred.add_argument("--model_dir", default="models",
                        help="Directory containing trained model .pkl files (default: models/).")
    p_pred.add_argument("--skip_transformer_sentiment", action="store_true",
                        help="Skip RuBERT transformer inference (faster but less accurate).")
    p_pred.add_argument("--posts_csv", default=None,
                        help="CSV with columns url, date, domain (one row per post). "
                             "When provided, enriches predictions with date/domain and "
                             "saves models/raw_counts_per_date.csv for the dashboard.")

    # ── build_dashboard ────────────────────────────────────────────────────
    p_dash = sub.add_parser("build_dashboard",
                            help="Merge post metadata with labelled comments for a dashboard.")
    p_dash.add_argument("--json_posts_folder", required=True,
                        help="Folder with JSON post-metadata files.")
    p_dash.add_argument("--csv_posts_folder", required=True,
                        help="Folder where converted CSV metadata files will be stored.")
    p_dash.add_argument("--final_df_folder", required=True,
                        help="Folder with processed/labelled comment CSVs.")
    p_dash.add_argument("--output_file", default="output_for_dash.csv",
                        help="Output path for the merged dashboard CSV.")

    # ── dashboard ──────────────────────────────────────────────────────────
    p_db = sub.add_parser("dashboard", help="Launch the Streamlit analytics dashboard.")
    p_db.add_argument("--input_file", default="predictions.csv",
                      help="Predictions CSV to visualise (default: predictions.csv).")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "convert":         run_convert,
        "label":           run_label,
        "train":           run_train,
        "predict":         run_predict,
        "build_dashboard": run_build_dashboard,
        "dashboard":       run_dashboard,
    }
    dispatch[args.mode](args)
