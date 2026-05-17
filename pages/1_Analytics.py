import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Analytics Dashboard", layout="wide")

st.title("📊 AI Ticket Analytics Dashboard")

log_file = "logs/predictions.csv"

# =========================
# LOAD DATA
# =========================
if os.path.exists(log_file):

    df = pd.read_csv(log_file)

    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # =========================
    # KPIs
    # =========================
    st.subheader("📌 Key Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Tickets", len(df))
    col2.metric("Avg Confidence", round(df["confidence"].mean() * 100, 2))
    col3.metric("Unique Categories", df["prediction"].nunique())

    st.divider()

    # =========================
    # MOST COMMON ISSUES
    # =========================
    st.subheader("🔥 Most Common Issues")

    issue_counts = df["prediction"].value_counts()

    st.bar_chart(issue_counts)

    # =========================
    # CATEGORY DISTRIBUTION
    # =========================
    st.subheader("📦 Category Distribution")

    st.bar_chart(issue_counts)

    # =========================
    # DAILY TICKETS
    # =========================
    st.subheader("📈 Daily Ticket Volume")

    df["date"] = df["time"].dt.date
    daily = df.groupby("date").size()

    st.line_chart(daily)

    # =========================
    # RAW DATA
    # =========================
    st.subheader("🧾 Latest Logs")

    st.dataframe(df.tail(20))

else:
    st.warning("No logs found yet. Run predictions first.")