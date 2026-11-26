# AI-Powered Customer Feedback Analyzer

A beginner-friendly Machine Learning, NLP, and Artificial Neural Network (ANN) capstone project that analyzes customer reviews/complaints and produces three predictions from a single feedback record:
1. **Sentiment**: `Positive`, `Neutral`, or `Negative`
2. **Complaint Type**: `Product Quality`, `Delivery`, `Technical Issue`, `Customer Service`, `Returns & Refunds`, `Billing`, `Pricing`, or `Packaging`
3. **Priority**: `Low`, `Medium`, or `High`

---

## Project Structure

```text
customer-feedback-analyzer/
├── data/
│   └── customer_feedback.csv
├── notebooks/
│   ├── data_analysis.ipynb
│   ├── nlp_models.ipynb
│   └── ann_model.ipynb
├── models/
│   ├── sentiment_pipeline.pkl
│   ├── complaint_model.pkl
│   ├── priority_ml_baseline.pkl
│   ├── priority_ann.keras
│   ├── priority_preprocessor.pkl
│   └── priority_encoder.pkl
├── train_all.py
├── app.py
├── requirements.txt
└── README.md
```

---

## Workflow Summary

1. **Data Understanding & EDA (`notebooks/data_analysis.ipynb`)**:
   - Audited 30,000 rows and 9 columns (0 missing values, 0 duplicates).
   - Visualized Sentiment, Complaint Type, Priority, Channel, and Customer Type distributions.
2. **NLP & Traditional ML (`notebooks/nlp_models.ipynb`)**:
   - Cleaned feedback text (lowercasing, punctuation/number removal).
   - Converted text into TF-IDF vectors.
   - Trained Logistic Regression models (with `GridSearchCV` hyperparameter tuning) for Sentiment (~100% test accuracy) and Complaint Type (~100% test accuracy), plus a traditional ML baseline for Priority (~48% test accuracy).
3. **Keras Artificial Neural Network (`notebooks/ann_model.ipynb`)**:
   - Built a feed-forward Keras ANN (`Dense(128, relu) -> Dropout -> Dense(64, relu) -> Dropout -> Dense(3, softmax)`) using `Adam` optimizer and `sparse_categorical_crossentropy` loss.
   - Compared ANN performance against the Logistic Regression baseline.
4. **Web Application (`app.py`)**:
   - Built a simple Streamlit interface where users enter customer feedback and get Sentiment, Complaint Type, and Priority predictions along with confidence scores.

---

## How to Run

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train and save all models** (either run the three notebooks in `notebooks/` OR run the helper script):
   ```bash
   python train_all.py
   ```

3. **Start the Streamlit web application**:
   ```bash
   streamlit run app.py
   ```
