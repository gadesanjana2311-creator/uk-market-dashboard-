
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from collections import Counter

st.set_page_config(
    page_title="UK Top 50 Music Analytics",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 UK Top 50 Music Analytics")
st.subheader("Market Structure & Artist Diversity Analysis")
st.markdown(
    "Explore artist concentration, chart rankings, "
    "collaborations, and music characteristics."
)

# -----------------------------
# 1. Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Atlantic_United_Kingdom.csv")
    df.columns = df.columns.str.strip().str.lower()

    required = [
        "date", "position", "song", "artist",
        "duration_ms", "album_type", "is_explicit"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["date"] = pd.to_datetime(
        df["date"], dayfirst=True, errors="coerce"
    )
    df["position"] = pd.to_numeric(
        df["position"], errors="coerce"
    )
    df["duration_ms"] = pd.to_numeric(
        df["duration_ms"], errors="coerce"
    )
    df["popularity"] = pd.to_numeric(
        df["popularity"], errors="coerce"
    ) if "popularity" in df.columns else np.nan

    df = df.dropna(
        subset=["date", "position", "song", "artist"]
    ).copy()

    for col in ["song", "artist", "album_type"]:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    df["is_explicit"] = (
        df["is_explicit"].astype(str).str.lower()
        .isin(["true", "1", "yes", "y", "explicit"])
    )

    # The source data separates artist names with &
    df["artist_list"] = df["artist"].apply(
        lambda x: list(dict.fromkeys(
            a.strip() for a in x.split("&") if a.strip()
        ))
    )

    df["duration_min"] = df["duration_ms"] / 60000
    df["is_collab"] = df["artist_list"].apply(
        lambda x: len(x) > 1
    )
    df["rank_group"] = np.where(
        df["position"] <= 10,
        "Top 10",
        "Positions 11–50"
    )

    return df


try:
    df = load_data()
except Exception as e:
    st.error(f"Unable to load dataset: {e}")
    st.stop()

# -----------------------------
# 2. Sidebar Filters
# -----------------------------
st.sidebar.header("Dashboard Filters")

min_date = df["date"].min().date()
max_date = df["date"].max().date()

date_range = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
    filtered = df[
        df["date"].dt.date.between(start_date, end_date)
    ].copy()
else:
    filtered = df.copy()

rank_options = ["All", "Top 10", "Positions 11–50"]

selected_rank = st.sidebar.selectbox(
    "Chart position group", rank_options
)

if selected_rank != "All":
    filtered = filtered[
        filtered["rank_group"] == selected_rank
    ].copy()

album_options = ["All"] + sorted(
    filtered["album_type"].dropna().unique().tolist()
)

selected_album = st.sidebar.selectbox(
    "Album type", album_options
)

if selected_album != "All":
    filtered = filtered[
        filtered["album_type"] == selected_album
    ].copy()

if filtered.empty:
    st.warning("No records match the selected filters.")
    st.stop()

# -----------------------------
# 3. Calculate KPIs
# -----------------------------
artist_counts = Counter(
    artist
    for artists in filtered["artist_list"]
    for artist in artists
)

total_artist_appearances = sum(artist_counts.values())
unique_artists = len(artist_counts)

top5_share = (
    sum(count for _, count in artist_counts.most_common(5))
    / total_artist_appearances * 100
    if total_artist_appearances else 0
)

diversity_score = (
    unique_artists / len(filtered) * 100
    if len(filtered) else 0
)

collab_rate = filtered["is_collab"].mean() * 100
explicit_rate = filtered["is_explicit"].mean() * 100
avg_duration = filtered["duration_min"].mean()

artist_shares = (
    pd.Series(artist_counts, dtype=float)
    / max(total_artist_appearances, 1)
)

hhi = (artist_shares ** 2).sum()

st.subheader("Key Performance Indicators")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Track Entries", f"{len(filtered):,}")
k2.metric("Unique Artists", f"{unique_artists:,}")
k3.metric("Top 5 Artist Share", f"{top5_share:.2f}%")
k4.metric("Collaboration Rate", f"{collab_rate:.2f}%")

k5, k6, k7, k8 = st.columns(4)
k5.metric("Explicit Content", f"{explicit_rate:.2f}%")
k6.metric("Average Duration", f"{avg_duration:.2f} min")
k7.metric("Artist Diversity Score", f"{diversity_score:.2f}%")
k8.metric("Artist HHI", f"{hhi:.4f}")

st.caption(
    "Diversity score = unique artists / filtered track entries. "
    "HHI = sum of squared artist appearance shares."
)

# -----------------------------
# 4. Artist Leaderboard
# -----------------------------
st.divider()
st.subheader("🏆 Top 15 Artists")

leaderboard = pd.DataFrame(
    artist_counts.most_common(15),
    columns=["Artist", "Appearances"]
)

fig = px.bar(
    leaderboard.sort_values("Appearances"),
    x="Appearances",
    y="Artist",
    orientation="h",
    text="Appearances",
    title="Artists by Individual Track Appearances"
)

fig.update_layout(yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 5. Rank Analysis
# -----------------------------
st.subheader("📊 Chart Position Analysis")

rank_summary = (
    filtered.groupby("rank_group", observed=False)
    .agg(
        Track_Count=("song", "size"),
        Average_Duration=("duration_min", "mean"),
        Collaboration_Rate=("is_collab", "mean"),
        Explicit_Rate=("is_explicit", "mean")
    )
    .reset_index()
)

rank_summary["Collaboration_Rate"] *= 100
rank_summary["Explicit_Rate"] *= 100

left, right = st.columns(2)

with left:
    fig = px.bar(
        rank_summary,
        x="rank_group",
        y="Collaboration_Rate",
        text_auto=".2f",
        title="Collaboration Rate by Rank Group"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.bar(
        rank_summary,
        x="rank_group",
        y="Explicit_Rate",
        text_auto=".2f",
        title="Explicit Content Rate by Rank Group"
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 6. Album Type Analysis
# -----------------------------
st.subheader("💿 Album Type Distribution")

album_summary = (
    filtered["album_type"].value_counts()
    .rename_axis("Album Type")
    .reset_index(name="Track Count")
)

left, right = st.columns(2)

with left:
    fig = px.pie(
        album_summary,
        names="Album Type",
        values="Track Count",
        hole=0.45,
        title="Distribution of Album Types"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.bar(
        album_summary,
        x="Album Type",
        y="Track Count",
        text="Track Count",
        title="Entries by Album Type"
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 7. Track Duration Analysis
# -----------------------------
st.subheader("⏱️ Track Duration Analysis")

duration_data = filtered.dropna(subset=["duration_min"])

fig = px.histogram(
    duration_data,
    x="duration_min",
    nbins=30,
    title="Distribution of Track Durations",
    labels={"duration_min": "Duration (minutes)"}
)
st.plotly_chart(fig, use_container_width=True)

if filtered["popularity"].notna().any():
    fig = px.scatter(
        filtered.dropna(
            subset=["duration_min", "popularity"]
        ),
        x="duration_min",
        y="popularity",
        hover_data=["song", "artist", "position"],
        title="Track Duration vs Popularity"
    )
    st.plotly_chart(fig, use_container_width=True)

    corr = filtered[
        ["duration_min", "popularity"]
    ].corr().iloc[0, 1]

    st.write(f"Duration-popularity correlation: {corr:.3f}")

# -----------------------------
# 8. Artist Appearance Table
# -----------------------------
st.subheader("🎤 Artist Appearance Details")

artist_table = leaderboard.copy()
st.dataframe(artist_table, use_container_width=True)

# -----------------------------
# 9. Filtered Dataset
# -----------------------------
st.subheader("📁 Filtered Dataset")

display_df = filtered.drop(
    columns=["artist_list"], errors="ignore"
)

st.dataframe(display_df, use_container_width=True)

csv_data = display_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Filtered CSV",
    data=csv_data,
    file_name="UK_Top_50_Filtered.csv",
    mime="text/csv"
)

st.success("Dashboard loaded successfully.")
