"""
make_calibration_sample.py — Calibration sample builder
========================================================
Creates a CSV for manual annotation from a raw JSON comments file.

Logic:
  1. Load all comments from the JSON file.
  2. For each cluster, select N comments that match its keyword filter
     (ensures the sample is informative, not purely random).
  3. Add M random comments for the "none" class.
  4. Shuffle and save — only id + text + hint_cls columns.
     The cluster and sentiment columns are left blank for manual filling.

After filling cluster and sentiment manually, use the file as:
  - Ground truth for  dev/evaluate_prompt.py --calibration_file <file>
  - Augmented training data for  dev/compare_llm_labelers.py --calibration_file <file>

Usage
-----
python dev/make_calibration_sample.py \\
    --input_file  "path/to/Raw json comments/ozon_comments.json" \\
    --output_file dev/calibration_sample.csv

python dev/make_calibration_sample.py \\
    --input_file  data/raw.json \\
    --output_file dev/calibration_sample.xlsx \\
    --n_per_class 30 \\
    --n_none      50
"""

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="Build a calibration sample for manual annotation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input_file", required=True,
                   help="Path to raw JSON comments file ({url: [comment, ...]} format).")
    p.add_argument("--output_file", default="dev/calibration_sample.csv",
                   help="Output path (.csv or .xlsx). Default: dev/calibration_sample.csv.")
    p.add_argument("--n_per_class", type=int, default=20,
                   help="Comments per cluster class (default: 20).")
    p.add_argument("--n_none", type=int, default=30,
                   help="Comments for the 'none' class (default: 30).")
    p.add_argument("--min_len", type=int, default=30,
                   help="Minimum comment length in characters (default: 30).")
    p.add_argument("--random_state", type=int, default=42,
                   help="Random seed (default: 42).")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Keyword filters (keep in sync with src/preprocessing.py and dev/compare_llm_labelers.py)
# ══════════════════════════════════════════════════════════════════════════════

FILTERS = {
    "chatbot": re.compile(
        r"чат.?бот|chat.?bot|бот.{0,15}(не отвеча|перенаправ|завис|цикл|повтор|шаблон)"
        r"|застря[лк].*бот|бот.*застря[лк]"
        r"|автоответ|робот отвеча|оператор недоступен|не могу дозвониться"
        r"|живой.{0,10}(оператор|человек)|переключ.{0,20}оператор",
        re.IGNORECASE,
    ),
    "delay": re.compile(
        r"доставк[аиу]|перенес|опоздал|не приехал|трек|отслежи|курьер|задержк"
        r"|не пришл|жду.*день|перенос.*дат|дата.*изменил"
        r"|не доставил|дата доставки|перенос.*заказ|заказ.*перенес",
        re.IGNORECASE,
    ),
    "pricing": re.compile(
        r"цена.*измени|цена.*поднял|цена.*выросл|стоимость.*измени"
        r"|подорожал|вздорожал|вчера стоил|утром.*цена|цена.*вечером"
        r"|несколько раз.*цен|цену.*подняли|ценообразовани"
        r"|цена.*стала.*дороже|стало.*дороже|было.*рублей.*стало"
        r"|скидка.{0,20}(убра|исчезл|пропал|удали|отмени)"
        r"|цена.{0,10}(поднялась|взлетела|скачет|прыгает|меняется)"
        r"|кэш.?бек.{0,20}(убра|исчезл|снизил|урезал|отмени)"
        r"|повысил.{0,10}цен|снял.{0,10}скидк",
        re.IGNORECASE,
    ),
    "recommendations": re.compile(
        r"рекоменда[цц].{0,30}(не работ|нерелевант|не те|плох|мусор|реклам|алгоритм)"
        r"|алгоритм.{0,30}(рекоменда|подбор|показ|выдач)"
        r"|нерелевантн.{0,30}(товар|реклам|подбор|рекоменда)"
        r"|реклам.{0,20}смешан.{0,20}(товар|рекоменда|подбор)"
        r"|\"для вас\"|похожие товары|вы смотрели|персонализац"
        r"|подборк[аи].{0,30}(реклам|нерелевант|не те|алгоритм|бесполезн)",
        re.IGNORECASE,
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = _parse_args()

    print(f"Loading {args.input_file} ...")
    with open(args.input_file, encoding="utf-8") as f:
        data = json.load(f)

    all_comments = [
        str(c).strip()
        for comments in data.values()
        for c in comments
        if c and len(str(c).strip()) >= args.min_len
    ]
    print(f"Total comments: {len(all_comments):,}")

    random.seed(args.random_state)
    random.shuffle(all_comments)

    # Select comments per cluster
    selected: dict = {}
    used = set()

    for cls, pattern in FILTERS.items():
        matched = [t for t in all_comments if pattern.search(t) and id(t) not in used]
        chosen  = random.sample(matched, min(args.n_per_class, len(matched)))
        selected[cls] = chosen
        used.update(id(t) for t in chosen)
        print(f"  {cls:<20} matched={len(matched):>6}  selected={len(chosen)}")

    # "none" — comments that match no cluster filter
    none_pool = [t for t in all_comments
                 if not any(p.search(t) for p in FILTERS.values()) and id(t) not in used]
    selected["none"] = random.sample(none_pool, min(args.n_none, len(none_pool)))
    print(f"  {'none':<20} pool={len(none_pool):>6}  selected={len(selected['none'])}")

    # Build DataFrame
    rows = []
    uid  = 1
    for cls, texts in selected.items():
        for text in texts:
            rows.append({
                "id":        uid,
                "comment":   text,
                "cluster":   "",     # fill manually
                "sentiment": "",     # fill manually
                "hint_cls":  cls,    # keyword-based hint — may be wrong!
            })
            uid += 1

    df = pd.DataFrame(rows).sample(frac=1, random_state=args.random_state).reset_index(drop=True)
    df["id"] = range(1, len(df) + 1)

    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".xlsx":
        df.to_excel(out, index=False)
    else:
        df.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"\nSaved {len(df)} rows -> {out}")
    print("\nFill in 'cluster' and 'sentiment' columns manually.")
    print("Valid values:")
    print("  cluster:   chatbot | delay | pricing | recommendations | none")
    print("  sentiment: positive | negative | neutral")
    print(f"\nThe 'hint_cls' column is keyword-based and may contain errors.")


if __name__ == "__main__":
    main()
