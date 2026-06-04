"""
ai_labeling.py — LLM-based cluster and sentiment labeling for marketplace comments.

Uses an OpenAI-compatible API (e.g. LiteLLM proxy, DeepSeek, Groq, OpenAI) to
assign each comment a *cluster* and a *sentiment* label.

Supported clusters:   chatbot | pricing | recommendations | delay | none
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

_PROMPT_TEMPLATE_EN = """\
You are analyzing customer comments about online marketplaces and e-commerce platforms.

For EACH message determine the cluster and sentiment.

━━━ CLUSTERS ━━━
- chatbot        — the message is about the SUPPORT BOT itself (automated/AI agent, chatbot, \
virtual assistant, canned auto-reply). The bot specifically must be the subject. \
NOT: complaints about a human operator, NOT general "support is bad" without mention of a bot.
- pricing        — the price of the SAME item changed within a short period \
(e.g. different price this morning vs now, price went up/down between visits, fluctuating). \
NOT: unauthorized charges, double charges, refunds, billing disputes, or general "expensive".
- recommendations — the message is about the RECOMMENDATION / SUGGESTION algorithm \
(the "for you" picks, suggested products, personalization). Can be NEGATIVE \
(irrelevant items, bad picks, suggesting already-bought products) OR POSITIVE \
(loving the suggestions, picks are spot on). NOT: general product quality complaints.
- delay          — user complains about a SPECIFIC order/package that is late, \
tracking not updating, courier didn't arrive, item still not delivered. \
NOT: general dissatisfaction where delivery is a side mention.
- none           — none of the above clusters apply (general complaint, question, praise, etc.)

━━━ SENTIMENT ━━━
- positive — satisfied, grateful, praising
- negative — dissatisfied, frustrated, complaining
- neutral  — neutral question, information request, ambiguous

━━━ RULES ━━━
- Use ONLY values from the lists above — no variations
- If unsure between two clusters — pick the more specific one
- Return ONLY a JSON array, no explanations, no markdown fences

━━━ EXAMPLES ━━━
[
  {{"id":1,"cluster":"chatbot","sentiment":"negative"}},
  {{"id":2,"cluster":"delay","sentiment":"negative"}},
  {{"id":3,"cluster":"pricing","sentiment":"negative"}},
  {{"id":4,"cluster":"recommendations","sentiment":"negative"}},
  {{"id":5,"cluster":"recommendations","sentiment":"positive"}},
  {{"id":6,"cluster":"none","sentiment":"negative"}},
  {{"id":7,"cluster":"none","sentiment":"positive"}},
  {{"id":8,"cluster":"delay","sentiment":"neutral"}}
]

// Example context:
// 1 → "Your chatbot keeps sending the same canned auto-reply. Useless bot."
// 2 → "My package hasn't arrived. Tracking says delivered but it's not here."
// 3 → "Same item was $29 this morning, now it's $45 a few hours later."
// 4 → "Your recommendations keep suggesting stuff I already bought. The algorithm is terrible."
// 5 → "Honestly your recommendations are spot on lately — I love everything you suggest!"
// 6 → "Received a completely broken item. Terrible quality."
// 7 → "Order arrived a day early, perfectly packed. Very happy!"
// 8 → "Can you update me on order #123-456? Tracking hasn't moved in 3 days."

━━━ MESSAGES TO LABEL ━━━
{comments}
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_batch(
    client: openai.OpenAI,
    batch: list,
    model: str,
    lang: str = "ru",
) -> Optional[list]:
    """Send one batch of comments to the LLM and return parsed predictions.

    Args:
        client: A configured :class:`openai.OpenAI` client instance.
        batch:  List of raw comment strings (keep ≤ 80 for best results).
        model:  Model identifier, e.g. ``"deepseek/deepseek-chat"``.
        lang:   Language for the prompt — ``"ru"`` (default) or ``"en"``.

    Returns:
        List of dicts ``{"id": int, "cluster": str, "sentiment": str}``,
        or ``None`` if the response could not be parsed.
    """
    template = _PROMPT_TEMPLATE_EN if lang == "en" else _PROMPT_TEMPLATE
    numbered = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(batch))
    prompt = template.format(comments=numbered)

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
    lang: str = "ru",
) -> pd.DataFrame:
    """Label an entire DataFrame with cluster and sentiment predictions.

    Each row in *df* gets two new columns:
    - ``predicted_cluster``   — one of chatbot/pricing/recommendations/delay/none
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
        lang:                    Language for the prompt — ``"ru"`` (default) or ``"en"``.

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
            results = classify_batch(client, batch, model, lang=lang)
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
