import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="NLP Analytics Dashboard",
    layout="wide"
)

# Load
@st.cache_data
def load_data():
    df = pd.read_parquet("data/semantic.parquet")

    df["text_length"] = df["processed_text"].str.len()

    return df

df = load_data()

# Sidebar
st.sidebar.title("Filters")

sentiments = st.sidebar.multiselect(
    "Sentiment",
    df["sentiment"].unique(),
    default=df["sentiment"].unique()
)

query = st.sidebar.text_input("Search text")

date_range = st.sidebar.date_input(
    "Date range",
    [df["date"].min(), df["date"].max()]
)

topic_cols = [c for c in df.columns if c.startswith("TOPIC_")]

selected_topics = st.sidebar.multiselect(
    "Topics",
    topic_cols
)

# Filter
filtered_df = df[df["sentiment"].isin(sentiments)]

if query:
    filtered_df = filtered_df[
        filtered_df["processed_text"].str.contains(query, na=False)
    ]

if len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df["date"] >= pd.to_datetime(date_range[0])) &
        (filtered_df["date"] <= pd.to_datetime(date_range[1]))
    ]

if selected_topics:
    mask = filtered_df[selected_topics].sum(axis=1) > 0
    filtered_df = filtered_df[mask]

# Header
st.title("NLP Analytics Dashboard")
st.caption("Real-time sentiment & topic analysis")

# KPI's
col1, col2, col3, col4 = st.columns(4)

col1.metric("Posts", len(filtered_df))
col2.metric("Avg Length", int(filtered_df["text_length"].mean()))

pos_ratio = (filtered_df["sentiment"] == "positive").mean()
col3.metric("Positive %", f"{pos_ratio:.1%}")

neg_ratio = (filtered_df["sentiment"] == "negative").mean()
col4.metric("Negative %", f"{neg_ratio:.1%}")

# Charts
colA, colB = st.columns(2)

with colA:
    fig = px.histogram(filtered_df, x="sentiment")
    st.plotly_chart(fig, use_container_width=True)

with colB:
    if "date" in filtered_df.columns:
        time_df = (
            filtered_df
            .groupby([filtered_df["date"].dt.date, "sentiment"])
            .size()
            .reset_index(name="count")
        )

        fig = px.line(time_df, x="date", y="count", color="sentiment")
        st.plotly_chart(fig, use_container_width=True)

# Topics
if topic_cols:
    st.subheader("Topics")

    topic_counts = {
        t: filtered_df[t].sum() for t in topic_cols
    }

    topic_df = pd.DataFrame({
        "topic": list(topic_counts.keys()),
        "count": list(topic_counts.values())
    }).sort_values("count", ascending=False)

    fig = px.bar(topic_df, x="topic", y="count")
    st.plotly_chart(fig, use_container_width=True)

# Posts
st.subheader("Posts")

st.dataframe(
    filtered_df[
        ["date", "sentiment", "processed_text"]
    ].head(100),
    use_container_width=True
)

# python -m streamlit run nlp_dashboard.py