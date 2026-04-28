import streamlit as st
import pandas as pd
import plotly.express as px

# Dashboard page config
st.set_page_config(page_title="NLP Analytics Dashboard", layout="wide")

# Loading data
@st.cache_data
def load_data():
    df = pd.read_parquet("data/processed/comments_enriched.parquet")

    # ensure datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # text length
    df["text_length"] = df["processed_text"].fillna("").str.len()

    # ensure lists
    for col in ["tags", "topics"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
    return df

df = load_data()

# Sidebar with filters
st.sidebar.title("Filters")

## Source filter
sources = sorted(df["source"].dropna().unique()) if "source" in df.columns else []
selected_sources = st.sidebar.multiselect("Source", options=sources, default=[])

## Brand filter
brands = sorted(df["brand"].dropna().unique()) if "brand" in df.columns else []
selected_brands = st.sidebar.multiselect("Brand", options=brands, default=[])

## Sentiment filter
sentiments = sorted(df["sentiment"].dropna().unique()) if "sentiment" in df.columns else []
selected_sentiments = st.sidebar.multiselect("Sentiment", options=sentiments, default=sentiments)

## Topics filter
topic_cols = [c for c in df.columns if c.startswith("TOPIC_")]
selected_topics = st.sidebar.multiselect("Topics",options=topic_cols)

## Text search
query = st.sidebar.text_input("Search text")

## Date filter
if "date" in df.columns:
    min_date = df["date"].min()
    max_date = df["date"].max()
    date_range = st.sidebar.date_input("Date Range", value=[min_date, max_date])
else:
    date_range = None

# Filter data
filtered_df = df.copy()

## Sentiment
if selected_sentiments:
    filtered_df = filtered_df[filtered_df["sentiment"].isin(selected_sentiments)]

## Source
if selected_sources:
    filtered_df = filtered_df[filtered_df["source"].isin(selected_sources)]

## Brand
if selected_brands:
    filtered_df = filtered_df[filtered_df["brand"].isin(selected_brands)]

## Text query
if query:
    filtered_df = filtered_df[filtered_df["processed_text"].str.contains(query, case=False, na=False)]

## Date range
if date_range and len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])
    filtered_df = filtered_df[(filtered_df["date"] >= start_date) & (filtered_df["date"] <= end_date)]

## Topics
if selected_topics:
    mask = filtered_df[selected_topics].sum(axis=1) > 0
    filtered_df = filtered_df[mask]

# Header
st.title("NLP Analytics Dashboard")
st.caption("Sentiment, topics, toxicity and semantic analysis")

# KPIs
total_posts = len(filtered_df)
avg_length = int(filtered_df["text_length"].mean()) if total_posts else 0
positive_ratio = (filtered_df["sentiment"] == "positive").mean() if total_posts else 0
negative_ratio = (filtered_df["sentiment"] == "negative").mean() if total_posts else 0

# Toxicity
toxicity_ratio = filtered_df["tags"].apply(lambda x: "TAG_PROFANITY" in x).mean() if total_posts else 0

# Sarcasm
sarcasm_ratio = filtered_df["tags"].apply(lambda x: "TAG_SARCASM" in x).mean() if total_posts else 0

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Posts", total_posts)
col2.metric("Avg Length", avg_length)
col3.metric("Positive %", f"{positive_ratio:.1%}")
col4.metric("Negative %", f"{negative_ratio:.1%}")
col5.metric("Toxic %", f"{toxicity_ratio:.1%}")

# Main charts
colA, colB = st.columns(2)

# Sentiment distribution
with colA:
    st.subheader("Sentiment Distribution")
    fig = px.histogram(filtered_df, x="sentiment")
    st.plotly_chart(fig, use_container_width=True)

# Sentiment over time
with colB:
    st.subheader("Sentiment Over Time")
    if "date" in filtered_df.columns and not filtered_df.empty:
        time_df = filtered_df.groupby([filtered_df["date"].dt.date, "sentiment"]).size().reset_index(name="count")
        fig = px.line(time_df, x="date", y="count", color="sentiment")
        st.plotly_chart(fig, use_container_width=True)

# Topics
if topic_cols:
    st.subheader("Topics")
    topic_counts = {topic: (filtered_df[topic].sum()) for topic in topic_cols}
    topic_df = pd.DataFrame({
        "topic": list(topic_counts.keys()),
        "count": list(topic_counts.values())
    })
    topic_df = topic_df.sort_values("count", ascending=False)
    fig = px.bar(topic_df, x="topic", y="count")
    st.plotly_chart(fig, use_container_width=True)

# Brand negativity
if "brand" in filtered_df.columns:
    st.subheader("Negative Sentiment by Brand")
    brand_negativity = filtered_df.groupby("brand")["sentiment"].apply(lambda x: (x == "negative").mean()).reset_index(name="negative_ratio")
    fig = px.bar(brand_negativity, x="brand", y="negative_ratio")
    st.plotly_chart(fig, use_container_width=True)

# An attempt at sarcasm analysis
st.subheader("Sarcasm Usage")
sarcasm_df = pd.DataFrame({
    "type": ["sarcasm", "non_sarcasm"],
    "count": [
        filtered_df["tags"].apply(lambda x:"TAG_SARCASM" in x).sum(),
        filtered_df["tags"].apply(lambda x:"TAG_SARCASM" not in x).sum()
    ]
})
fig = px.pie(sarcasm_df, names="type", values="count")
st.plotly_chart(fig, use_container_width=True)

# Posts table
st.subheader("Posts")
display_columns = [col for col in [
        "date",
        "source",
        "brand",
        "sentiment",
        "sentiment_confidence",
        "processed_text"
    ] if col in filtered_df.columns
]
st.dataframe(filtered_df[display_columns].head(200), use_container_width=True)

# Raw data
with st.expander("Show raw dataframe"): st.dataframe(filtered_df, use_container_width=True)

# Footer
st.caption(f"Loaded {len(df)} total records")