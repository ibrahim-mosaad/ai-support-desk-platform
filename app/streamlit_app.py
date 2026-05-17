import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd
from datetime import datetime

# Load model
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# =========================
# Text Cleaning
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return " ".join(text.split())

# =========================
# UI
# =========================
st.set_page_config(page_title="AI Ticket Router", layout="centered")

st.title("💼 AI Customer Support System (Production)")

# =========================
# MULTI INPUT FORM (IMPORTANT UPGRADE)
# =========================

subject = st.text_input("Ticket Subject")
description = st.text_area("Ticket Description")
ticket_type = st.selectbox(
    "Ticket Type",
    ["Technical Issue", "Billing", "Account Access", "Product Inquiry"]
)

product = st.text_input("Product Purchased")
channel = st.selectbox(
    "Ticket Channel",
    ["Email", "Phone", "Chat", "Social Media"]
)

# =========================
# Predict
# =========================
if st.button("Predict Category"):

    combined_text = f"{subject} {description} {ticket_type} {product} {channel}"
    cleaned = clean_text(combined_text)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]
    proba = model.predict_proba(vector).max()

    # Output
    st.success(f"📌 Predicted Category: {prediction}")
    st.info(f"🎯 Confidence Score: {round(proba * 100, 2)}%")

    # =========================
    # Logging (Production)
    # =========================
    log_data = {
        "time": datetime.now(),
        "subject": subject,
        "description": description,
        "type": ticket_type,
        "product": product,
        "channel": channel,
        "prediction": prediction,
        "confidence": proba
    }

    df_log = pd.DataFrame([log_data])

    import os
    if not os.path.exists("logs/predictions.csv"):
        df_log.to_csv("logs/predictions.csv", index=False)
    else:
        df_log.to_csv("logs/predictions.csv", mode='a', header=False, index=False)

else:
    st.warning("Please fill required fields")