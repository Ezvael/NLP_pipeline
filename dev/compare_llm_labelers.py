"""
compare_llm_labelers.py — LLM labeler comparison tool
======================================================
Labels a random sample from a raw JSON comments file with one or more LLM
models, trains LR + LinearSVC on each set of labels, and compares macro F1.

Supports resumable overnight runs via per-model CSV checkpoints.

Usage
-----
# Label with the default model (DeepSeek), evaluate, save results:
python dev/compare_llm_labelers.py \\
    --input_file  "path/to/Raw json comments/ozon_comments.json" \\
    --config      config.json \\
    --output_file dev/llm_labeler_comparison.csv

# Use a different model and a larger sample:
python dev/compare_llm_labelers.py \\
    --input_file  data/raw.json \\
    --config      config.json \\
    --models      "deepseek/deepseek-chat" "openai/gpt-4o-mini" \\
    --sample_size 1000

# Append a manually annotated calibration file for augmented training:
python dev/compare_llm_labelers.py \\
    --input_file       data/raw.json \\
    --config           config.json \\
    --calibration_file dev/calibration_sample.xlsx
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import openai
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

# Make sure the project root is on sys.path so src.* imports work.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing import lemmatize_new

# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="Compare LLM labelers for cluster/sentiment classification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input_file", required=True,
                   help="Path to raw JSON comments file ({url: [comment, ...]} format).")
    p.add_argument("--config", default="config.json",
                   help="Path to config.json with api_key, base_url (default: config.json).")
    p.add_argument("--models", nargs="+", default=["deepseek/deepseek-v4-pro"],
                   help="List of model identifiers to compare (default: deepseek/deepseek-v4-pro).")
    p.add_argument("--sample_size", type=int, default=500,
                   help="Number of comments to sample from the regex-filtered pool (default: 500).")
    p.add_argument("--calibration_file", default=None,
                   help="Path to calibration_sample.xlsx with columns: comment, cluster, sentiment.")
    p.add_argument("--output_file", default="dev/llm_labeler_comparison.csv",
                   help="Where to save the comparison results CSV (default: dev/llm_labeler_comparison.csv).")
    p.add_argument("--checkpoint_dir", default=None,
                   help="Directory for per-model checkpoint CSVs (default: same folder as input_file).")
    p.add_argument("--batch_size", type=int, default=25,
                   help="Comments per LLM call (default: 25).")
    p.add_argument("--random_state", type=int, default=42,
                   help="Random seed (default: 42).")
    p.add_argument("--min_len", type=int, default=30,
                   help="Minimum comment length to include in sample (default: 30).")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Regex filters (canonical version — keep in sync with src/preprocessing.py)
# ══════════════════════════════════════════════════════════════════════════════

_FILTERS = {
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
# Prompt (keep in sync with src/ai_labeling.py _PROMPT_TEMPLATE)
# ══════════════════════════════════════════════════════════════════════════════

_PROMPT = """\
Ты анализируешь отзывы пользователей о маркетплейсах (Ozon, Wildberries и др.).

Для КАЖДОГО сообщения определи cluster и sentiment.

━━━ КЛАСТЕРЫ ━━━
- chatbot        — пользователь напрямую взаимодействует с чат-ботом поддержки: бот не отвечает, застрял в боте, бот перенаправляет в круг. НЕ подходит: слово "бот" употреблено в переносном смысле ("ботом не отделаетесь", "без ботов и накруток")
- pricing        — цена на ОДИН И ТОТ ЖЕ товар заметно изменилась за короткий период (часы/дни): пользователь увидел другую цену сегодня vs вчера/утром/вечером, добавил в корзину по одной цене — оформил по другой. НЕ подходит: региональные различия цен, общее недовольство ценами без привязки к конкретному изменению, ответы поддержки с объяснением ценовой политики
- recommendations — пользователь говорит о работе алгоритмов рекомендаций: нерелевантные товары в подборке, реклама смешана с рекомендациями, персонализация не работает. НЕ подходит: упоминание "подборки" без критики алгоритма
- delay          — пользователь жалуется на КОНКРЕТНУЮ задержку или перенос даты доставки своего заказа, трек не обновляется, курьер не приехал. НЕ подходит: общее недовольство сервисом где доставка упомянута вскользь, ответы поддержки с объяснением задержки
- none           — ни один из вышеперечисленных кластеров не подходит

━━━ СЕНТИМЕНТ ━━━
- positive — доволен, хвалит
- negative — недоволен, жалуется
- neutral  — нейтрально, вопрос, информация

━━━ ПРАВИЛА ━━━
- Используй ТОЛЬКО значения из списков выше, никаких вариаций
- Если сомневаешься между двумя кластерами — выбери более специфичный
- Верни ТОЛЬКО JSON-список, без пояснений и markdown

━━━ ПРИМЕРЫ (реальные отзывы) ━━━
[
  {{"id":1,"cluster":"chatbot","sentiment":"negative"}},   // "Деньги не возвращают при отмене заказа и бот не отвечает. Куда писать?"
  {{"id":2,"cluster":"delay","sentiment":"negative"}},     // "Курьер не приехал, трек завис, перенесли уже третий раз подряд"
  {{"id":3,"cluster":"pricing","sentiment":"negative"}},   // "Утром цена 1200, вечером 1800 — так нельзя, это обман покупателей"
  {{"id":4,"cluster":"recommendations","sentiment":"negative"}}, // "Рекомендации смешаны с рекламой, невозможно найти нужное"
  {{"id":5,"cluster":"none","sentiment":"negative"}},      // "Не советую покупать, привезли побитую мебель и деньги не вернули"
  {{"id":6,"cluster":"none","sentiment":"positive"}},      // "Отличный сервис, всё пришло вовремя, рекомендую!"
  {{"id":7,"cluster":"delay","sentiment":"neutral"}}       // "Подскажите, когда обновится статус доставки по трек-номеру?"
]

━━━ СООБЩЕНИЯ ДЛЯ РАЗМЕТКИ ━━━
{comments}
"""

# ══════════════════════════════════════════════════════════════════════════════
# Label normalisation
# ══════════════════════════════════════════════════════════════════════════════

VALID_CLUSTERS   = {"none", "delay", "chatbot", "pricing", "recommendations"}
VALID_SENTIMENTS = {"positive", "negative", "neutral"}

_CLUSTER_MAP = {
    "no_cluster": "none", "no cluster": "none", "other": "none",
    "nothing": "none", "general": "none", "unrelated": "none",
    "delivery": "delay", "shipping": "delay", "timing": "delay",
    "logistics": "delay", "доставка": "delay",
    "bot": "chatbot", "chat_bot": "chatbot", "support_bot": "chatbot", "chat bot": "chatbot",
    "price": "pricing", "cost": "pricing", "цена": "pricing",
    "recommendation": "recommendations", "recommend": "recommendations",
    "рекомендации": "recommendations",
}
_SENTIMENT_MAP = {
    "pos": "positive", "neg": "negative", "neu": "neutral", "mixed": "neutral",
    "нейтральный": "neutral", "положительный": "positive", "отрицательный": "negative",
}


def _normalize_cluster(raw: str) -> str:
    s = raw.strip().lower()
    return s if s in VALID_CLUSTERS else _CLUSTER_MAP.get(s, raw.strip())


def _normalize_sentiment(raw: str) -> str:
    s = raw.strip().lower()
    return s if s in VALID_SENTIMENTS else _SENTIMENT_MAP.get(s, raw.strip())


# ══════════════════════════════════════════════════════════════════════════════
# JSON extraction helper
# ══════════════════════════════════════════════════════════════════════════════

def _extract_json(text: str) -> Optional[list]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None


def _safe_name(model: str) -> str:
    return re.sub(r"[^\w\-]", "_", model)


# ══════════════════════════════════════════════════════════════════════════════
# Labeling (with mid-model checkpoints)
# ══════════════════════════════════════════════════════════════════════════════

MAX_RETRIES      = 3
RETRY_SLEEP      = 10
CHECKPOINT_EVERY = 4   # save checkpoint every N batches


def label_with_model(comments: list, model: str, client: openai.OpenAI,
                     ckpt_dir: Path, batch_size: int, log) -> pd.DataFrame:
    safe      = _safe_name(model)
    done_path = ckpt_dir / f"labeled_{safe}.csv"
    part_path = ckpt_dir / f"labeled_{safe}.partial.csv"

    done_rows: list = []
    start_idx = 0
    if part_path.exists():
        partial   = pd.read_csv(part_path, encoding="utf-8-sig")
        done_rows = partial.to_dict("records")
        start_idx = len(done_rows)
        log.info(f"  [{model}] Resuming from partial checkpoint: {start_idx}/{len(comments)}")

    batches = [comments[i:i+batch_size] for i in range(start_idx, len(comments), batch_size)]
    n_total = len(comments)
    n_done_batches = start_idx // batch_size
    t_start = time.time()

    for b_idx, batch in enumerate(batches):
        n = len(batch)
        clusters   = ["error"] * n
        sentiments = ["error"] * n

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(batch))
                resp = client.chat.completions.create(
                    model=model, temperature=0,
                    messages=[{"role": "user", "content": _PROMPT.format(comments=numbered)}],
                    timeout=120,
                )
                parsed = _extract_json(resp.choices[0].message.content or "")
                if parsed:
                    for item in parsed:
                        idx = item.get("id", 0) - 1
                        if 0 <= idx < n:
                            clusters[idx]   = _normalize_cluster(str(item.get("cluster",   "error")))
                            sentiments[idx] = _normalize_sentiment(str(item.get("sentiment", "error")))
                    missing = [i for i in range(n) if clusters[i] == "error"]
                    if missing and attempt < MAX_RETRIES:
                        log.info(f"  [{model}] {len(missing)} missing ids, retrying...")
                        r2 = client.chat.completions.create(
                            model=model, temperature=0,
                            messages=[{"role": "user", "content": _PROMPT.format(
                                comments="\n".join(f"{i+1}. {batch[i]}" for i in missing)
                            )}],
                            timeout=120,
                        )
                        p2 = _extract_json(r2.choices[0].message.content or "")
                        if p2:
                            for j, item2 in enumerate(p2):
                                orig = missing[j] if j < len(missing) else -1
                                if 0 <= orig < n and clusters[orig] == "error":
                                    clusters[orig]   = _normalize_cluster(str(item2.get("cluster",   "error")))
                                    sentiments[orig] = _normalize_sentiment(str(item2.get("sentiment", "error")))
                break
            except Exception as exc:
                log.warning(f"  [{model}] batch {n_done_batches+b_idx+1} attempt {attempt}: {exc}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_SLEEP * attempt)

        for i, text in enumerate(batch):
            done_rows.append({
                "comment":             text,
                "predicted_cluster":   clusters[i],
                "predicted_sentiment": sentiments[i],
            })

        n_done_batches += 1
        elapsed   = time.time() - t_start
        rate      = len(done_rows) / elapsed if elapsed > 0 else 0
        remaining = (n_total - len(done_rows)) / rate if rate > 0 else 0
        log.info(f"  [{model}] rows {len(done_rows)}/{n_total}  "
                 f"elapsed {elapsed/60:.1f}m  eta {remaining/60:.1f}m")

        if b_idx > 0 and (b_idx + 1) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(done_rows).to_csv(part_path, index=False, encoding="utf-8-sig")
            log.info(f"  [{model}] Partial checkpoint saved ({len(done_rows)} rows)")

    df = pd.DataFrame(done_rows)
    df.to_csv(done_path, index=False, encoding="utf-8-sig")
    if part_path.exists():
        part_path.unlink()

    error_n = (df["predicted_cluster"] == "error").sum()
    log.info(f"  [{model}] DONE  errors={error_n}  dist={df['predicted_cluster'].value_counts().to_dict()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ML evaluation
# ══════════════════════════════════════════════════════════════════════════════

def eval_labeler(lemmatized: pd.Series, labels: pd.Series, task: str,
                 model_name: str, random_state: int, log,
                 extra_texts: pd.Series = None, extra_labels: pd.Series = None):
    counts = labels.value_counts()
    valid  = counts[counts >= 2].index
    mask   = labels.isin(valid)
    texts  = lemmatized[mask].reset_index(drop=True)
    y_raw  = labels[mask].reset_index(drop=True)

    if len(y_raw.unique()) < 2:
        log.warning(f"  [{model_name}|{task}] Not enough classes, skipping.")
        return []

    le = LabelEncoder()
    y  = le.fit_transform(y_raw)

    Xtr_all, Xte, ytr_all, yte = train_test_split(
        texts, y, test_size=0.3, stratify=y, random_state=random_state
    )
    Xtr, Xval, ytr, yval = train_test_split(
        Xtr_all, ytr_all, test_size=0.5, stratify=ytr_all, random_state=random_state
    )

    if extra_texts is not None and extra_labels is not None and len(extra_texts) > 0:
        extra_valid = extra_labels.isin(le.classes_)
        ex_t = extra_texts[extra_valid].reset_index(drop=True)
        ex_y = le.transform(extra_labels[extra_valid].reset_index(drop=True))
        Xtr  = pd.concat([Xtr.reset_index(drop=True), ex_t], ignore_index=True)
        ytr  = list(ytr) + list(ex_y)
        log.info(f"  [{model_name}|{task}] +{extra_valid.sum()} calibration rows added to train")

    vec   = TfidfVectorizer()
    Xtr_v = vec.fit_transform(Xtr)
    Xva_v = vec.transform(Xval)
    Xte_v = vec.transform(Xte)

    results = []
    for clf_name, clf in [
        ("LR",  LogisticRegression(solver="saga", C=0.5, fit_intercept=False,
                                   max_iter=2000, class_weight="balanced",
                                   random_state=random_state)),
        ("SVM", LinearSVC(C=0.25, tol=0.001, max_iter=5000,
                          class_weight="balanced", random_state=random_state)),
    ]:
        clf.fit(Xtr_v, ytr)
        f1_tr = f1_score(ytr,  clf.predict(Xtr_v), average="macro")
        f1_va = f1_score(yval, clf.predict(Xva_v), average="macro")
        f1_te = f1_score(yte,  clf.predict(Xte_v), average="macro")

        report = classification_report(yte, clf.predict(Xte_v),
                                       target_names=le.classes_, digits=3, zero_division=0)
        log.info(f"\n  [{model_name} | {task} | {clf_name}]  "
                 f"train={f1_tr:.4f}  valid={f1_va:.4f}  test={f1_te:.4f}\n{report}")

        results.append({
            "llm_model": model_name,
            "task":      task,
            "clf":       clf_name,
            "f1_train":  round(f1_tr, 4),
            "f1_valid":  round(f1_va, 4),
            "f1_test":   round(f1_te, 4),
        })
    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = _parse_args()

    # Logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"llm_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger(__name__)
    log.info(f"Log: {log_file}")

    # Config
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    client = openai.OpenAI(api_key=cfg["api_key"], base_url=cfg.get("base_url", ""))

    # Checkpoint dir
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else Path(args.input_file).parent / "llm_comparison_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # STEP 1: Load & sample
    log.info("STEP 1 — Loading comments")
    with open(args.input_file, encoding="utf-8") as f:
        data = json.load(f)

    all_comments = [
        str(c).strip()
        for comments in data.values()
        for c in comments
        if c and len(str(c).strip()) >= args.min_len
    ]
    log.info(f"Total: {len(all_comments):,}")

    random.seed(args.random_state)
    filtered = [c for c in all_comments if any(p.search(c) for p in _FILTERS.values())]
    log.info(f"Regex-filtered pool: {len(filtered):,}")
    sample = random.sample(filtered, min(args.sample_size, len(filtered)))
    log.info(f"Sampled: {len(sample):,}")

    # STEP 2: Label
    log.info("STEP 2 — LLM labeling")
    labeled: dict = {}
    for model in args.models:
        done_path = ckpt_dir / f"labeled_{_safe_name(model)}.csv"
        if done_path.exists():
            log.info(f"[SKIP] {model} (checkpoint exists)")
            labeled[model] = pd.read_csv(done_path, encoding="utf-8-sig")
        else:
            log.info(f"[RUN] {model}")
            labeled[model] = label_with_model(
                sample, model, client, ckpt_dir, args.batch_size, log
            )
            time.sleep(2)

    # STEP 3: Lemmatize
    log.info("STEP 3 — Lemmatizing")
    t = time.time()
    lemmatized = pd.Series(sample).astype(str).apply(lemmatize_new)
    log.info(f"Done in {time.time()-t:.1f}s")

    # STEP 3b: Calibration data
    calib_lem  = pd.Series(dtype=str)
    calib_cls  = pd.Series(dtype=str)
    calib_sent = pd.Series(dtype=str)
    if args.calibration_file and Path(args.calibration_file).exists():
        log.info(f"Loading calibration data from {args.calibration_file}")
        df_calib = pd.read_excel(args.calibration_file)
        df_calib.columns = [c.strip() for c in df_calib.columns]
        df_calib["comment"]   = df_calib["comment"].fillna("").astype(str)
        df_calib["cluster"]   = df_calib["cluster"].str.strip().str.lower()
        df_calib["sentiment"] = df_calib["sentiment"].str.strip().str.lower()
        df_calib = df_calib[
            df_calib["cluster"].isin(VALID_CLUSTERS) &
            df_calib["sentiment"].isin(VALID_SENTIMENTS)
        ].reset_index(drop=True)
        calib_lem  = df_calib["comment"].apply(lemmatize_new)
        calib_cls  = df_calib["cluster"]
        calib_sent = df_calib["sentiment"]
        log.info(f"Calibration: {len(df_calib)} rows, dist={calib_cls.value_counts().to_dict()}")

    # STEP 4: Evaluate
    log.info("STEP 4 — Training & evaluating")
    all_results = []
    for model, df in labeled.items():
        df  = df.reset_index(drop=True)
        lem = lemmatized.reset_index(drop=True)
        df_c = df[df["predicted_cluster"].isin(VALID_CLUSTERS)].copy()
        lem_c = lem[df_c.index]
        for task, col, ex_lbl in [
            ("cluster",   "predicted_cluster",   calib_cls),
            ("sentiment", "predicted_sentiment", calib_sent),
        ]:
            rows = eval_labeler(
                lem_c.reset_index(drop=True),
                df_c[col].reset_index(drop=True),
                task, model, args.random_state, log,
                extra_texts=calib_lem.reset_index(drop=True) if len(calib_lem) > 0 else None,
                extra_labels=ex_lbl.reset_index(drop=True)   if len(ex_lbl)  > 0 else None,
            )
            all_results.extend(rows)

    # STEP 5: Summary
    df_res = pd.DataFrame(all_results)
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    df_res.to_csv(args.output_file, index=False, encoding="utf-8-sig")
    log.info(f"Results saved -> {args.output_file}")

    for task in ["cluster", "sentiment"]:
        sub   = df_res[df_res["task"] == task]
        pivot = sub.pivot_table(index="llm_model", columns="clf",
                                values="f1_test", aggfunc="first")
        pivot.columns = [f"f1_test_{c}" for c in pivot.columns]
        pivot = pivot.sort_values(pivot.columns[0], ascending=False)
        log.info(f"\n--- {task.upper()} f1_test ---\n{pivot.round(4).to_string()}")


if __name__ == "__main__":
    main()
