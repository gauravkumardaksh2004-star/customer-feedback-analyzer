import os
import re
import string
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras


def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


st.set_page_config(page_title="Customer Feedback Analyzer", page_icon="💬")

st.title("CUSTOMER FEEDBACK ANALYZER")
st.write("Enter a customer review or complaint to predict **Sentiment**, **Complaint Type**, and **Priority**.")

# Check if saved models exist
model_files = [
    "models/sentiment_pipeline.pkl",
    "models/complaint_model.pkl",
    "models/priority_ann.keras",
    "models/priority_preprocessor.pkl",
    "models/priority_encoder.pkl",
]

if not all(os.path.exists(f) for f in model_files):
    st.error("Saved models not found! Please run `python train_all.py` (or run the notebooks) first.")
    st.stop()


@st.cache_resource
def load_models():
    sentiment_model = joblib.load("models/sentiment_pipeline.pkl")
    complaint_model = joblib.load("models/complaint_model.pkl")
    priority_ann = keras.models.load_model("models/priority_ann.keras")
    priority_prep = joblib.load("models/priority_preprocessor.pkl")
    priority_encoder = joblib.load("models/priority_encoder.pkl")
    return sentiment_model, complaint_model, priority_ann, priority_prep, priority_encoder


sentiment_model, complaint_model, priority_ann, priority_prep, priority_encoder = load_models()

# User Input Form
feedback_text = st.text_area(
    "Enter customer feedback:",
    placeholder="Example: The product arrived two days late and the package was badly damaged. I also had difficulty getting support to respond.",
    height=120
)

with st.expander("Optional Customer Details (used by Priority ANN model)"):
    col1, col2 = st.columns(2)
    with col1:
        channel = st.selectbox("Channel", ["Website", "Mobile App", "Email", "Phone", "In-Store"])
        order_value = st.number_input("Order Value ($)", min_value=0.0, value=55.0)
    with col2:
        customer_type = st.selectbox("Customer Type", ["Returning", "New", "Loyal"])
        days_as_customer = st.number_input("Days as Customer", min_value=1, value=350)

if st.button("ANALYZE"):
    if feedback_text.strip() == "":
        st.warning("Please enter some customer feedback first.")
    else:
        cleaned = clean_text(feedback_text)

        # 1. Sentiment Prediction
        sentiment_pred = sentiment_model.predict([cleaned])[0]
        sentiment_conf = np.max(sentiment_model.predict_proba([cleaned])) * 100

        # 2. Complaint Type Prediction
        complaint_pred = complaint_model.predict([cleaned])[0]
        complaint_conf = np.max(complaint_model.predict_proba([cleaned])) * 100

        # 3. Priority Prediction (Keras ANN)
        input_data = pd.DataFrame([{
            "clean_text": cleaned,
            "channel": channel,
            "customer_type": customer_type,
            "order_value": order_value,
            "days_as_customer": days_as_customer
        }])
        X_input = priority_prep.transform(input_data).toarray()
        prio_probs = priority_ann.predict(X_input, verbose=0)[0]
        prio_index = np.argmax(prio_probs)
        priority_pred = priority_encoder.inverse_transform([prio_index])[0]
        priority_conf = prio_probs[prio_index] * 100

        # Display Output (Matching Capstone PDF Section 7)
        st.subheader("Prediction Results")
        st.write(f"**Sentiment:** {sentiment_pred} *(Confidence: {sentiment_conf:.1f}%)*")
        st.write(f"**Complaint Type:** {complaint_pred} *(Confidence: {complaint_conf:.1f}%)*")
        st.write(f"**Priority:** {priority_pred} *(Confidence: {priority_conf:.1f}%)*")
