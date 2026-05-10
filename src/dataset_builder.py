"""
dataset_builder.py — Build the final merged dataset ready for a dashboard.

Joins post-level metadata (domain, date, url) with the processed and labelled
comment DataFrames to produce a single flat table.

Typical usage
-------------
>>> from src.dataset_builder import build_dashboard_dataset
>>> df_dash = build_dashboard_dataset(
...     json_posts_folder="data/json_with_posts",
...     csv_posts_folder="data/csv_with_posts",
...     final_df_folder="data/final_df",
...     output_path="output_for_dash.csv",
... )
"""

import os

import pandas as pd

from src.data_loader import combine_csv_files, json_posts_folder_to_csv


# ---------------------------------------------------------------------------
# Core merge logic
# ---------------------------------------------------------------------------

def merge_metadata_with_comments(
    metadata_df: pd.DataFrame,
    processed_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join post-level metadata with a processed comments DataFrame on URL.

    The *processed_df* is expected to have columns produced by the full
    pipeline (``AgentPlatform.ipynb`` → ``Prepare full df.ipynb`` flow),
    including at minimum ``Ссылка`` (or ``url``), ``comment`` (or ``Комментарий``),
    ``new comment``, ``cluster``, and ``sentiment``.

    Args:
        metadata_df:  DataFrame with columns
                      ``[url, domain, id_group, id_post, date]``.
        processed_df: DataFrame with processed comment data.

    Returns:
        Merged DataFrame with columns:
        ``[domain, url, date, cluster, sentiment, comment, new_comment]``.
        Only rows where *url* appears in both inputs are kept (inner join).
    """
    # Normalise column names coming from different pipeline stages
    rename_map = {
        "Ссылка": "url",
        "Комментарий": "comment",
        "new comment": "new_comment",
    }
    df2 = processed_df.rename(columns={k: v for k, v in rename_map.items() if k in processed_df.columns})

    merged = pd.merge(metadata_df, df2, on="url", how="inner")

    desired_columns = ["domain", "url", "date", "cluster", "sentiment", "comment", "new_comment"]
    available = [c for c in desired_columns if c in merged.columns]
    result = merged[available]

    print(f"[dataset_builder] Merged {len(result)} rows (metadata × comments)")
    return result


# ---------------------------------------------------------------------------
# End-to-end builder
# ---------------------------------------------------------------------------

def build_dashboard_dataset(
    json_posts_folder: str,
    csv_posts_folder: str,
    final_df_folder: str,
    output_path: str = "output_for_dash.csv",
) -> pd.DataFrame:
    """End-to-end pipeline: JSON metadata → CSV → merge with comments → save.

    Steps performed:
    1. Convert all JSON post-metadata files in *json_posts_folder* to CSV
       files in *csv_posts_folder*.
    2. Combine all CSV files in *csv_posts_folder* into one metadata DataFrame.
    3. Combine all CSV files in *final_df_folder* into one comments DataFrame.
    4. Merge metadata and comments on ``url``.
    5. Save the result to *output_path*.

    Args:
        json_posts_folder: Folder with JSON post-metadata files
                           (format: url → {domain, id_group, id_post, date}).
        csv_posts_folder:  Folder where converted CSV metadata files are stored.
                           Created automatically if it does not exist.
        final_df_folder:   Folder with processed / labelled comment CSV files
                           (output of the full prediction pipeline).
        output_path:       Path for the final merged CSV.

    Returns:
        The merged :class:`pandas.DataFrame`.
    """
    # Step 1: convert JSON metadata → CSV
    json_posts_folder_to_csv(json_posts_folder, csv_posts_folder)

    # Step 2: load metadata
    metadata_df = combine_csv_files(csv_posts_folder)

    # Step 3: load processed comments
    processed_df = combine_csv_files(final_df_folder)

    # Step 4: merge
    result = merge_metadata_with_comments(metadata_df, processed_df)

    # Step 5: save
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"[dataset_builder] Dashboard dataset saved to '{output_path}'")

    return result
