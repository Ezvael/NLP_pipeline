"""
ai_labeling.py — LLM-based cluster and sentiment labeling for marketplace comments.

Uses an OpenAI-compatible API (e.g. LiteLLM proxy, DeepSeek, Groq, OpenAI) to
assign each comment a *cluster* and a *sentiment* label.

Supported clusters:   chatbot | pricing | recommendations | suggest | delay | none
Supported sentiments: positive | negative | neutral

Typical usage
-------------
>>> import json
>>> from src.ai_labeling import label_dataset
>>> from src.data_loader import load_dataset
>>>
>>> with open("config.json") as f:
...     cfg = json.load(f)
>>>
>>> df = load_dataset("data/raw_datasets/ozon_comments.csv")
>>> df_labeled = label_dataset(
...     df,
...     api_key=cfg["api_key"],
...     base_url=cfg["base_url"],
...     model=cfg["model"],
... )
>>> df_labeled.to_csv("data/labeled/ozon_labeled.csv", index=False)
"""

import json
import re
import time
from typing import Optional

import openai
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chunks(lst: list, n: int):
    """Yield successive *n*-sized chunks from *lst*."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _extract_json(text: str) -> Optional[list]:
    """Extract the first JSON array found anywhere in *text*.

    LLMs sometimes wrap the JSON in markdown fences or add a short
    explanation before/after the array — this function strips that noise.
    """
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
Ты анализируешь отзывы к маркетплейсам.

Для КАЖДОГО сообщения определи:

cluster:
- chatbot        — в тексте упоминается общение с ботом поддержки или чат-ботом
- pricing        — пользователи упоминают изменение цены по несколько раз в течение часа/дня/недели
- recommendations — пользователи упоминают в тексте работу алгоритмов рекомендации
- suggest        — пользователи упоминают поисковые подсказчики (саджесты)
- delay          — пользователи упоминают работу сервиса отслеживания времени доставки
- none           — ни один из вышеперечисленных кластеров не подходит

sentiment:
- positive
- negative
- neutral

КРИТИЧНО: используй только кластеры и сантименты из списка выше.
Return ONLY a JSON list, no extra text.

Example:
[
  {{"id":1,"cluster":"delay","sentiment":"negative"}},
  {{"id":2,"cluster":"chatbot","sentiment":"positive"}}
]

Messages:
{comments}
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_batch(
    client: openai.OpenAI,
    batch: list,
    model: str,
) -> Optional[list]:
    """Send one batch of comments to the LLM and return parsed predictions.

    Args:
        client: A configured :class:`openai.OpenAI` client instance.
        batch:  List of raw comment strings (keep ≤ 80 for best results).
        model:  Model identifier, e.g. ``"deepseek/deepseek-chat"``.

    Returns:
        List of dicts ``{"id": int, "cluster": str, "sentiment": str}``,
        or ``None`` if the response could not be parsed.
    """
    numbered = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(batch))
    prompt = _PROMPT_TEMPLATE.format(comments=numbered)

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return _extract_json(response.choices[0].message.content or "")


def label_dataset(
    df: pd.DataFrame,
    api_key: str,
    base_url: str,
    model: str,
    text_column: str = "comment",
    batch_size: int = 80,
    sleep_between_batches: float = 0.5,
) -> pd.DataFrame:
    """Label an entire DataFrame with cluster and sentiment predictions.

    Each row in *df* gets two new columns:
    - ``predicted_cluster``   — one of chatbot/pricing/recommendations/suggest/delay/none
    - ``predicted_sentiment`` — one of positive/negative/neutral

    Rows that fail (API error, parse error) are marked as ``"error"``.

    Args:
        df:                      DataFrame containing *text_column*.
        api_key:                 API key for the LLM provider.
        base_url:                Base URL of the OpenAI-compatible endpoint.
                                 E.g. ``"https://litellm.tokengate.ru/v1"``.
        model:                   Model identifier string passed to the API.
        text_column:             Column with raw comment text.
        batch_size:              Number of comments per API call.
        sleep_between_batches:   Seconds to wait between requests (rate limiting).

    Returns:
        Copy of *df* with ``predicted_cluster`` and ``predicted_sentiment`` columns.
    """
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    comments_list = df[text_column].fillna("").tolist()

    predicted_clusters = ["error"] * len(comments_list)
    predicted_sentiments = ["error"] * len(comments_list)

    batches = list(_chunks(comments_list, batch_size))
    for batch_num, batch in enumerate(tqdm(batches, desc="[ai_labeling] Labeling")):
        global_offset = batch_num * batch_size
        try:
            results = classify_batch(client, batch, model)
            if results:
                for item in results:
                    idx = global_offset + item["id"] - 1
                    if 0 <= idx < len(comments_list):
                        predicted_clusters[idx] = item.get("cluster", "error")
                        predicted_sentiments[idx] = item.get("sentiment", "error")
        except Exception as exc:
            print(f"[ai_labeling] Batch {batch_num} failed: {exc}")

        time.sleep(sleep_between_batches)

    df = df.copy()
    df["predicted_cluster"] = predicted_clusters
    df["predicted_sentiment"] = predicted_sentiments
    return df
