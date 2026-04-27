import argparse
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="NLP Analytics Dashboard", layout="wide")


def parse_input_file() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input_file", default="predictions.csv")
    args, _ = parser.parse_known_args()
    return args.input_file


@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path, encoding='utf-8-sig')

    # Normalise text column: accept 'comment' as alias for 'Комментарий'
    if "Комментарий" not in df.columns and "comment" in df.columns:
        df = df.rename(columns={"comment": "Комментарий"})

    # Prefer transformer sentiment (when it has actual values);
    # fall back to predicted_ml_sentiment, then to the original LLM label
    for fallback in ("predicted_ml_sentiment", "sentiment_mode", "sentiment"):
        if df.get("transformer_sentiment") is None or df["transformer_sentiment"].isna().all():
            if fallback in df.columns:
                df["transformer_sentiment"] = df[fallback]
                break

    # Prefer predicted_cluster; fall back to the existing cluster column
    if "predicted_cluster" not in df.columns:
        for fallback in ("cluster_mode", "cluster"):
            if fallback in df.columns:
                df["predicted_cluster"] = df[fallback]
                break

    df["text_length"] = df["Комментарий"].astype(str).str.len()

    if "date" in df.columns:
        # Handle Unix timestamps (int) as well as date strings
        if pd.api.types.is_integer_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"], unit="s", errors="coerce")
        else:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


input_file = parse_input_file()
df = load_data(input_file)

# Sidebar
st.sidebar.title("Filters")

sentiments = st.sidebar.multiselect(
    "Sentiment",
    options=sorted(df["transformer_sentiment"].dropna().unique()),
    default=sorted(df["transformer_sentiment"].dropna().unique()),
)

selected_domains = []
if "domain" in df.columns:
    domains = sorted(df["domain"].dropna().unique())
    selected_domains = st.sidebar.multiselect("Domain", domains)

selected_clusters = []
if "predicted_cluster" in df.columns:
    clusters = sorted(df["predicted_cluster"].dropna().unique())
    selected_clusters = st.sidebar.multiselect("Cluster", clusters)

date_range = []
if "date" in df.columns and df["date"].notna().any():
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    date_range = st.sidebar.date_input("Date range", [min_date, max_date])

query = st.sidebar.text_input("Search text")

# Filter
filtered = df[df["transformer_sentiment"].isin(sentiments)]

if selected_domains:
    filtered = filtered[filtered["domain"].isin(selected_domains)]

if selected_clusters:
    filtered = filtered[filtered["predicted_cluster"].isin(selected_clusters)]

if len(date_range) == 2 and "date" in filtered.columns:
    filtered = filtered[
        (filtered["date"] >= pd.to_datetime(date_range[0]))
        & (filtered["date"] <= pd.to_datetime(date_range[1]))
    ]

if query:
    filtered = filtered[
        filtered["Комментарий"].astype(str).str.contains(query, na=False, case=False)
    ]

# Header
st.title("NLP Analytics Dashboard")
st.caption(f"Source: `{input_file}` — {len(filtered):,} posts shown")

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Posts", f"{len(filtered):,}")
col2.metric("Avg Length", int(filtered["text_length"].mean()) if len(filtered) else 0)
pos_pct = (filtered["transformer_sentiment"] == "positive").mean()
neg_pct = (filtered["transformer_sentiment"] == "negative").mean()
col3.metric("Positive %", f"{pos_pct:.1%}")
col4.metric("Negative %", f"{neg_pct:.1%}")

# Charts row 1
colA, colB = st.columns(2)

with colA:
    st.subheader("Sentiment distribution")
    fig = px.histogram(
        filtered,
        x="transformer_sentiment",
        color="transformer_sentiment",
        color_discrete_map={"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with colB:
    if "date" in filtered.columns and filtered["date"].notna().any():
        st.subheader("Sentiment over time")
        time_df = (
            filtered.groupby([filtered["date"].dt.date, "transformer_sentiment"])
            .size()
            .reset_index(name="count")
        )
        fig = px.line(
            time_df,
            x="date",
            y="count",
            color="transformer_sentiment",
            color_discrete_map={"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"},
        )
        st.plotly_chart(fig, use_container_width=True)
    elif "domain" in filtered.columns:
        st.subheader("Posts by domain")
        domain_df = filtered["domain"].value_counts().reset_index()
        domain_df.columns = ["domain", "count"]
        fig = px.bar(domain_df, x="domain", y="count")
        st.plotly_chart(fig, use_container_width=True)

# Charts row 2 — clusters
if "predicted_cluster" in filtered.columns:
    st.subheader("Cluster distribution")
    cluster_df = filtered["predicted_cluster"].value_counts().reset_index()
    cluster_df.columns = ["cluster", "count"]
    fig = px.bar(cluster_df, x="cluster", y="count", color="cluster")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# Posts table
st.subheader("Posts")
table_cols = [c for c in ["date", "domain", "Комментарий", "transformer_sentiment",
                           "transformer_sentiment_confidence", "predicted_cluster"]
              if c in filtered.columns]
st.dataframe(filtered[table_cols].head(200), use_container_width=True)

# python -m streamlit run dashboard.py
