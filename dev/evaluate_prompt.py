"""
evaluate_prompt.py — Prompt calibration tool
============================================
Runs the canonical LLM prompt on a manually annotated calibration file and
reports per-class precision / recall / F1 for both cluster and sentiment,
plus a confusion matrix and the top-15 cluster errors.

Usage
-----
python dev/evaluate_prompt.py --calibration_file dev/calibration_sample.xlsx
python dev/evaluate_prompt.py --calibration_file dev/calibration_sample.xlsx \\
    --model deepseek/deepseek-v4-pro
python dev/evaluate_prompt.py --calibration_file dev/calibration_sample.xlsx \\
    --config path/to/config.json
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import openai
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

sys.stdout.reconfigure(encoding="utf-8")

# Make sure the project root is on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate the LLM prompt against a manually annotated calibration file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--calibration_file", required=True,
                   help="Path to calibration file (.xlsx or .csv) with columns: "
                        "comment, cluster, sentiment.")
    p.add_argument("--config", default="config.json",
                   help="Path to config.json with api_key and base_url (default: config.json).")
    p.add_argument("--model", default="moonshotai/kimi-k2",
                   help="LLM model identifier (default: moonshotai/kimi-k2).")
    p.add_argument("--batch_size", type=int, default=25,
                   help="Comments per API call (default: 25).")
    p.add_argument("--output_file", default=None,
                   help="Optional path to save full results as Excel (default: auto-named).")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

VALID_CLUSTERS   = {"none", "delay", "chatbot", "pricing", "recommendations"}
VALID_SENTIMENTS = {"positive", "negative", "neutral"}

MAX_RETRIES = 3
RETRY_SLEEP = 10

# ── Label normalisation ───────────────────────────────────────────────────────

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


def _norm_c(raw: str) -> str:
    s = str(raw).strip().lower()
    return s if s in VALID_CLUSTERS else _CLUSTER_MAP.get(s, "error")


def _norm_s(raw: str) -> str:
    s = str(raw).strip().lower()
    return s if s in VALID_SENTIMENTS else _SENTIMENT_MAP.get(s, "error")


# ══════════════════════════════════════════════════════════════════════════════
# Prompt (must stay in sync with src/ai_labeling.py and dev/compare_llm_labelers.py)
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


def _extract_json(text: str) -> Optional[list]:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            return None
    return None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = _parse_args()

    # Load config
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    client = openai.OpenAI(api_key=cfg["api_key"], base_url=cfg.get("base_url", ""))

    # Load calibration data
    calib_path = Path(args.calibration_file)
    if calib_path.suffix == ".xlsx":
        df = pd.read_excel(calib_path)
    else:
        df = pd.read_csv(calib_path, encoding="utf-8-sig")

    df.columns    = [c.strip() for c in df.columns]
    df["comment"] = df["comment"].fillna("").astype(str)
    df["cluster"] = df["cluster"].str.strip().str.lower()
    df["sentiment"] = df["sentiment"].str.strip().str.lower()
    df = df[
        df["cluster"].isin(VALID_CLUSTERS) &
        df["sentiment"].isin(VALID_SENTIMENTS)
    ].reset_index(drop=True)
    n = len(df)

    print(f"Model:            {args.model}")
    print(f"Rows:             {n}")
    print(f"Cluster dist:     {df['cluster'].value_counts().to_dict()}")
    print(f"Sentiment dist:   {df['sentiment'].value_counts().to_dict()}")

    # Run LLM labeling
    comments   = df["comment"].tolist()
    llm_c_raw  = ["error"] * n
    llm_s_raw  = ["error"] * n
    llm_c_norm = ["error"] * n
    llm_s_norm = ["error"] * n

    batches = [list(range(i, min(i + args.batch_size, n))) for i in range(0, n, args.batch_size)]
    print(f"\nRunning {len(batches)} batches...")

    for b_idx, idxs in enumerate(batches):
        batch  = [comments[i] for i in idxs]
        nb     = len(batch)
        c_buf  = ["error"] * nb
        s_buf  = ["error"] * nb
        cr_buf = ["error"] * nb
        sr_buf = ["error"] * nb

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(batch))
                resp = client.chat.completions.create(
                    model=args.model, temperature=0,
                    messages=[{"role": "user", "content": _PROMPT.format(comments=numbered)}],
                    timeout=120,
                )
                parsed = _extract_json(resp.choices[0].message.content or "")
                if parsed:
                    for item in parsed:
                        pos = item.get("id", 0) - 1
                        if 0 <= pos < nb:
                            c_buf[pos]  = str(item.get("cluster",   "error"))
                            s_buf[pos]  = str(item.get("sentiment", "error"))
                            cr_buf[pos] = _norm_c(c_buf[pos])
                            sr_buf[pos] = _norm_s(s_buf[pos])
                    missing = [pos for pos in range(nb) if cr_buf[pos] == "error"]
                    if missing and attempt < MAX_RETRIES:
                        print(f"  batch {b_idx+1}: retrying {len(missing)} missing ids...")
                        r2 = client.chat.completions.create(
                            model=args.model, temperature=0,
                            messages=[{"role": "user", "content": _PROMPT.format(
                                comments="\n".join(f"{i+1}. {batch[pos]}" for i, pos in enumerate(missing))
                            )}],
                            timeout=120,
                        )
                        p2 = _extract_json(r2.choices[0].message.content or "")
                        if p2:
                            for j, item2 in enumerate(p2):
                                pos = missing[j] if j < len(missing) else -1
                                if 0 <= pos < nb and cr_buf[pos] == "error":
                                    c_buf[pos]  = str(item2.get("cluster",   "error"))
                                    s_buf[pos]  = str(item2.get("sentiment", "error"))
                                    cr_buf[pos] = _norm_c(c_buf[pos])
                                    sr_buf[pos] = _norm_s(s_buf[pos])
                break
            except Exception as e:
                print(f"  batch {b_idx+1} attempt {attempt}/{MAX_RETRIES}: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_SLEEP * attempt)

        for k, global_i in enumerate(idxs):
            llm_c_raw[global_i]  = c_buf[k]
            llm_s_raw[global_i]  = s_buf[k]
            llm_c_norm[global_i] = cr_buf[k]
            llm_s_norm[global_i] = sr_buf[k]

        errs = cr_buf.count("error")
        print(f"  batch {b_idx+1}/{len(batches)}  errors={errs}/{nb}")

    df["llm_cluster_raw"]   = llm_c_raw
    df["llm_sentiment_raw"] = llm_s_raw
    df["llm_cluster"]       = llm_c_norm
    df["llm_sentiment"]     = llm_s_norm

    SEP = "=" * 65

    # Invalid rate
    n_err_c_raw  = sum(v not in VALID_CLUSTERS   for v in llm_c_raw)
    n_err_s_raw  = sum(v not in VALID_SENTIMENTS for v in llm_s_raw)
    n_err_c_norm = llm_c_norm.count("error")
    n_err_s_norm = llm_s_norm.count("error")
    print(f"\n{SEP}")
    print(f"  INVALID LABEL RATE  (model={args.model})")
    print(SEP)
    print(f"  cluster   before fuzzy: {n_err_c_raw}/{n} ({n_err_c_raw/n:.1%})   after: {n_err_c_norm}/{n} ({n_err_c_norm/n:.1%})")
    print(f"  sentiment before fuzzy: {n_err_s_raw}/{n} ({n_err_s_raw/n:.1%})   after: {n_err_s_norm}/{n} ({n_err_s_norm/n:.1%})")

    # Cluster report
    df_c = df[df["llm_cluster"] != "error"].copy()
    cls_labels = sorted(VALID_CLUSTERS)
    print(f"\n{SEP}")
    print(f"  CLUSTER  (evaluated on {len(df_c)}/{n} rows)")
    print(SEP)
    print(classification_report(df_c["cluster"], df_c["llm_cluster"],
                                 labels=cls_labels, zero_division=0, digits=3))
    print("  Confusion matrix (rows=human, cols=LLM):")
    cm = confusion_matrix(df_c["cluster"], df_c["llm_cluster"], labels=cls_labels)
    print(pd.DataFrame(cm, index=cls_labels, columns=cls_labels).to_string())

    # Sentiment report
    df_s = df[df["llm_sentiment"] != "error"].copy()
    sent_labels = ["negative", "neutral", "positive"]
    print(f"\n{SEP}")
    print(f"  SENTIMENT  (evaluated on {len(df_s)}/{n} rows)")
    print(SEP)
    print(classification_report(df_s["sentiment"], df_s["llm_sentiment"],
                                 labels=sent_labels, zero_division=0, digits=3))
    print("  Confusion matrix (rows=human, cols=LLM):")
    cm2 = confusion_matrix(df_s["sentiment"], df_s["llm_sentiment"], labels=sent_labels)
    print(pd.DataFrame(cm2, index=sent_labels, columns=sent_labels).to_string())

    # Top cluster errors
    print(f"\n{SEP}")
    print("  TOP CLUSTER ERRORS")
    print(SEP)
    errs = df_c[df_c["cluster"] != df_c["llm_cluster"]][["comment", "cluster", "llm_cluster"]]
    for _, r in errs.head(15).iterrows():
        print(f"  human={r['cluster']:<18} llm={r['llm_cluster']:<18} | {str(r['comment'])[:85]}")

    # Save full results
    out = args.output_file or str(calib_path.parent / f"calib_eval_{args.model.replace('/', '_')}.xlsx")
    df.to_excel(out, index=False)
    print(f"\nFull results saved -> {out}")
    print("DONE")


if __name__ == "__main__":
    main()
