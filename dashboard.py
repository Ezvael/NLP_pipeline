import argparse
import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="NLP Analytics Dashboard", layout="wide")

def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input_file", default=os.path.join(BASE_DIR, "data", "all_predictions.csv.gz"))
    parser.add_argument("--raw_counts",
                        default=os.path.join(BASE_DIR, "data", "all_raw_counts.csv"),
                        help="CSV with columns: date, domain, total_comments (raw counts)")
    args, _ = parser.parse_known_args()
    return args

def parse_input_file() -> str:
    return parse_args().input_file


@st.cache_data
def load_data(path: str):
    # Accept both plain CSV and .gz compressed CSV
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path = path + ".gz"
    if not os.path.exists(path):
        st.error(f"Data file not found: `{path}`\n\n"
                 "Run `python run_full_test.py` first to generate predictions.csv, "
                 "or pass a different path via `--input_file`.")
        st.stop()
    enc = None if path.endswith(".gz") else "utf-8-sig"
    df = pd.read_csv(path, encoding=enc)

    # Normalise text column — accept several aliases for 'comment'
    for alias in ("Комментарий", "Kommentarij"):
        if "comment" not in df.columns and alias in df.columns:
            df = df.rename(columns={alias: "comment"})
            break

    # Prefer transformer sentiment (when it has actual values);
    # fall back to predicted_ml_sentiment, then to the original LLM label
    for fallback in ("predicted_sentiment", "predicted_ml_sentiment", "sentiment_mode", "sentiment"):
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

    if "comment" not in df.columns:
        st.error("No comment column found in dataset.")
        st.stop()
        
    df["text_length"] = df["comment"].astype(str).str.len()

    # Enrich with post dates + domain from ozon_comments.csv when input has Ссылка but no date
    if "date" not in df.columns and "Ссылка" in df.columns:
        _url_date_path = os.path.join(BASE_DIR, "data", "ozon_comments.csv")
        if os.path.exists(_url_date_path):
            _ud = pd.read_csv(_url_date_path, usecols=["url", "date", "domain"])
            df = df.merge(_ud.rename(columns={"url": "Ссылка"}), on="Ссылка", how="left")

    # Domain fallback: extract from VK URL
    if "domain" not in df.columns and "Ссылка" in df.columns:
        df["domain"] = (
            df["Ссылка"].astype(str)
            .str.extract(r"vk\.com/([^/_?#]+)", expand=False)
        )

    if "date" in df.columns:
        # Handle Unix timestamps (int) as well as date strings
        if pd.api.types.is_integer_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"], unit="s", errors="coerce")
        else:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


_args      = parse_args()
input_file = _args.input_file
df = load_data(input_file)

# Load pre-computed raw comment counts per date (saved by run_full_test.py)
@st.cache_data
def load_raw_counts(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    rc = pd.read_csv(path)
    rc["date"] = pd.to_datetime(rc["date"], errors="coerce")
    return rc.dropna(subset=["date"]).sort_values("date")

df_raw_counts = load_raw_counts(_args.raw_counts)

# Sidebar
st.sidebar.title("Filters")

sentiments = st.sidebar.multiselect(
    "Sentiment",
    options=sorted(df["transformer_sentiment"].dropna().unique()),
    default=sorted(df["transformer_sentiment"].dropna().unique()),
)

selected_sources = []
if "source" in df.columns:
    sources = sorted(df["source"].dropna().unique())
    selected_sources = st.sidebar.multiselect("Source", sources)

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

if selected_sources:
    filtered = filtered[filtered["source"].isin(selected_sources)]

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
        filtered["comment"].astype(str).str.contains(query, na=False, case=False)
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
    st.plotly_chart(fig, width="stretch")

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
        st.plotly_chart(fig, width="stretch")
    elif "domain" in filtered.columns:
        st.subheader("Posts by domain")
        domain_df = filtered["domain"].value_counts().reset_index()
        domain_df.columns = ["domain", "count"]
        fig = px.bar(domain_df, x="domain", y="count")
        st.plotly_chart(fig, width="stretch")

# Charts row 2 — clusters
if "predicted_cluster" in filtered.columns:
    st.subheader("Cluster distribution")
    cluster_df = filtered["predicted_cluster"].value_counts().reset_index()
    cluster_df.columns = ["cluster", "count"]
    fig = px.bar(cluster_df, x="cluster", y="count", color="cluster")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

# Cluster share over time (cluster counts from filtered / raw total per date)
if (
    df_raw_counts is not None
    and "predicted_cluster" in filtered.columns
    and "date" in filtered.columns
    and filtered["date"].notna().any()
):
    st.subheader("Cluster mentions over time")

    CLUSTER_COLORS_SHARE = {
        "delay":           "#4C78A8",
        "chatbot":         "#F58518",
        "pricing":         "#E45756",
        "recommendations": "#54A24B",
    }
    target_clusters = list(CLUSTER_COLORS_SHARE.keys())

    # Granularity selector
    granularity = st.selectbox(
        "Period",
        options=["Week", "Month", "Quarter"],
        index=1,
        key="share_granularity",
    )
    freq_map = {"Week": "W", "Month": "M", "Quarter": "Q"}
    freq = freq_map[granularity]

    # Cluster counts per period from filtered df
    filt_copy = filtered[filtered["predicted_cluster"].isin(target_clusters)].copy()
    filt_copy["_period"] = filt_copy["date"].dt.to_period(freq).dt.to_timestamp()
    cluster_per_period = (
        filt_copy.groupby(["_period", "predicted_cluster"])
        .size()
        .reset_index(name="count")
        .rename(columns={"_period": "period", "predicted_cluster": "cluster"})
    )

    # Raw totals aggregated to same period.
    # df_raw_counts columns: date, domain, total_comments.
    # Apply source filter first (maps source → domains), then domain filter.
    SOURCE_DOMAIN_MAP = {
        "reddit": {"amazon", "amazonprime"},
        "vk":     {"ozon", "aliexpress", "megamrkt", "yandex.market", "wildberries"},
    }
    rc_src = df_raw_counts.copy()
    if selected_sources and "domain" in rc_src.columns:
        allowed = set()
        for s in selected_sources:
            allowed |= SOURCE_DOMAIN_MAP.get(s, set())
        rc_src = rc_src[rc_src["domain"].isin(allowed)]
    if selected_domains and "domain" in rc_src.columns:
        rc_src = rc_src[rc_src["domain"].isin(selected_domains)]
    rc_src["period"] = rc_src["date"].dt.to_period(freq).dt.to_timestamp()
    rc_period = rc_src.groupby("period")["total_comments"].sum().reset_index(name="total")

    if not cluster_per_period.empty:
        import plotly.graph_objects as go

        fig_overlay = go.Figure()
        for cluster, color in CLUSTER_COLORS_SHARE.items():
            sub = cluster_per_period[cluster_per_period["cluster"] == cluster]
            fig_overlay.add_trace(go.Bar(
                x=sub["period"], y=sub["count"],
                name=cluster, marker_color=color,
                yaxis="y1",
            ))
        fig_overlay.add_trace(go.Scatter(
            x=rc_period["period"], y=rc_period["total"],
            name="Total comments", mode="lines+markers",
            line=dict(color="#6C757D", width=2, dash="dot"),
            marker=dict(size=4),
            yaxis="y2",
        ))
        fig_overlay.update_layout(
            title=f"Cluster mentions vs. total volume ({granularity.lower()})",
            barmode="group", bargap=0.15,
            xaxis=dict(title="Period"),
            yaxis=dict(title="Cluster comments", side="left"),
            yaxis2=dict(title="Total comments", side="right", overlaying="y", showgrid=False),
            hovermode="x unified",
            legend=dict(orientation="h", y=1.05, x=0),
        )
        st.plotly_chart(fig_overlay, width="stretch")
    else:
        st.info("Not enough data to build the chart.")

# Posts table
st.subheader("Posts")
table_cols = [c for c in ["date", "domain", "comment", "transformer_sentiment",
                           "transformer_sentiment_confidence", "predicted_cluster", "tags"]
              if c in filtered.columns]
st.dataframe(filtered[table_cols].head(200), width="stretch")

# python -m streamlit run dashboard.py
