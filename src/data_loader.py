"""
data_loader.py — Raw data loading and format conversion utilities.

Handles:
  - Converting raw JSON comment files → CSV
  - Converting JSON post-metadata files → CSV
  - Combining multiple CSV files from a folder into one DataFrame
  - Loading and validating a dataset CSV

Typical usage
-------------
>>> from src.data_loader import json_to_csv, combine_csv_files, load_dataset
>>> json_to_csv("data/raw/ozon_comments.json", "data/raw_datasets/ozon_comments.csv")
>>> df = load_dataset("data/raw_datasets/ozon_comments.csv")
"""

import csv
import json
import os

import pandas as pd


# ---------------------------------------------------------------------------
# JSON → CSV helpers
# ---------------------------------------------------------------------------

def json_to_csv(input_file: str, output_file: str) -> None:
    """Convert a single raw JSON comments file to a flat CSV.

    The input JSON format is::

        {
          "https://some-url/...": ["comment text 1", "comment text 2"],
          ...
        }

    The resulting CSV has two columns: ``url`` and ``comment``.

    Args:
        input_file:  Path to the source ``.json`` file.
        output_file: Path where the ``.csv`` file will be written.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "comment"])
        for url, comments in data.items():
            for comment in comments:
                writer.writerow([url, comment])

    print(f"[data_loader] Saved {output_file} ({sum(len(v) for v in data.values())} rows)")


def json_posts_folder_to_csv(input_folder: str, output_folder: str) -> None:
    """Convert a folder of JSON post-metadata files to individual CSV files.

    Each JSON file maps URL → ``{domain, id_group, id_post, date}``.
    One CSV is produced per JSON file, with the same base name.

    Args:
        input_folder:  Directory containing ``.json`` metadata files.
        output_folder: Directory where converted ``.csv`` files will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)

    converted = 0
    for filename in os.listdir(input_folder):
        if not filename.endswith(".json"):
            continue

        json_path = os.path.join(input_folder, filename)
        csv_path = os.path.join(output_folder, filename.replace(".json", ".csv"))

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        rows = [
            {
                "url": url,
                "domain": values.get("domain"),
                "id_group": values.get("id_group"),
                "id_post": values.get("id_post"),
                "date": values.get("date"),
            }
            for url, values in data.items()
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["url", "domain", "id_group", "id_post", "date"]
            )
            writer.writeheader()
            writer.writerows(rows)

        converted += 1

    print(f"[data_loader] Converted {converted} JSON files → '{output_folder}'")


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def combine_csv_files(folder_path: str) -> pd.DataFrame:
    """Read and vertically concatenate all ``.csv`` files in a folder.

    Files are sorted alphabetically before merging so the result is
    deterministic across runs.

    Args:
        folder_path: Directory that contains ``.csv`` files.

    Returns:
        A single :class:`pandas.DataFrame` with a reset integer index.

    Raises:
        FileNotFoundError: If no CSV files are found in *folder_path*.
    """
    files = sorted(f for f in os.listdir(folder_path) if f.endswith(".csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in '{folder_path}'")

    frames = [pd.read_csv(os.path.join(folder_path, f)) for f in files]
    combined = pd.concat(frames, ignore_index=True)
    print(f"[data_loader] Combined {len(files)} CSV files from '{folder_path}' → {len(combined)} rows")
    return combined


def load_dataset(filepath: str, text_column: str = "comment") -> pd.DataFrame:
    """Load a CSV dataset and validate that the required text column exists.

    Args:
        filepath:    Path to the ``.csv`` file.
        text_column: Name of the column that contains raw comment text.
                     Defaults to ``"comment"``.

    Returns:
        The loaded :class:`pandas.DataFrame`.

    Raises:
        ValueError: If *text_column* is not present in the file.
    """
    df = pd.read_csv(filepath)
    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found in '{filepath}'. "
            f"Available columns: {list(df.columns)}"
        )
    print(f"[data_loader] Loaded '{filepath}' — {len(df)} rows")
    return df
