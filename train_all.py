import os
import re
import string
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    os.makedirs("models", exist_ok=True)

    print("1. Loading dataset...")
    df = pd.read_csv("data/customer_feedback.csv")
    df = df.drop_duplicates()
    df["clean_text"] = df["feedback_text"].apply(clean_text)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["priority"]
    )

    # ---------------------------------------------------------
    # Task 1: Sentiment Model (TF-IDF + Tuned Logistic Regression)
    # ---------------------------------------------------------
    print("\n2. Training Sentiment Analysis Model...")
    sent_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("model", LogisticRegression(max_iter=1000))
    ])
    grid_sent = GridSearchCV(
        sent_pipe,
        {"tfidf__max_features": [2000, 3000], "model__C": [0.5, 1.0]},
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )
    grid_sent.fit(train_df["clean_text"], train_df["sentiment"])
    best_sent = grid_sent.best_estimator_
    print("Sentiment Accuracy:", round(accuracy_score(test_df["sentiment"], best_sent.predict(test_df["clean_text"])), 4))
    joblib.dump(best_sent, "models/sentiment_pipeline.pkl")

    # ---------------------------------------------------------
    # Task 2: Complaint Classification Model
    # ---------------------------------------------------------
    print("\n3. Training Complaint Classification Model...")
    comp_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
        ("model", LogisticRegression(max_iter=1000))
    ])
    comp_pipe.fit(train_df["clean_text"], train_df["complaint_type"])
    print("Complaint Type Accuracy:", round(accuracy_score(test_df["complaint_type"], comp_pipe.predict(test_df["clean_text"])), 4))
    joblib.dump(comp_pipe, "models/complaint_model.pkl")

    # ---------------------------------------------------------
    # Task 3A: Priority Traditional ML Baseline
    # ---------------------------------------------------------
    print("\n4. Training Priority Traditional ML Baseline...")
    feature_cols = ["clean_text", "channel", "customer_type", "order_value", "days_as_customer"]
    preprocessor = ColumnTransformer([
        ("text", TfidfVectorizer(max_features=1500, ngram_range=(1, 2)), "clean_text"),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["channel", "customer_type"]),
        ("num", StandardScaler(), ["order_value", "days_as_customer"])
    ])
    prio_ml = Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ])
    prio_ml.fit(train_df[feature_cols], train_df["priority"])
    print("Priority ML Baseline Accuracy:", round(accuracy_score(test_df["priority"], prio_ml.predict(test_df[feature_cols])), 4))
    joblib.dump(prio_ml, "models/priority_ml_baseline.pkl")

    # ---------------------------------------------------------
    # Task 3B: Priority Keras ANN
    # ---------------------------------------------------------
    print("\n5. Training Priority Keras ANN Model...")
    ann_prep = ColumnTransformer([
        ("text", TfidfVectorizer(max_features=1500, ngram_range=(1, 2)), "clean_text"),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["channel", "customer_type"]),
        ("num", StandardScaler(), ["order_value", "days_as_customer"])
    ])
    X_train = ann_prep.fit_transform(train_df[feature_cols]).toarray()
    X_test = ann_prep.transform(test_df[feature_cols]).toarray()

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["priority"])
    y_test = le.transform(test_df["priority"])

    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(3, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=15,
        batch_size=128,
        callbacks=[early_stop],
        verbose=1
    )

    y_pred_ann = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("Priority Keras ANN Accuracy:", round(accuracy_score(y_test, y_pred_ann), 4))
    print(classification_report(y_test, y_pred_ann, target_names=le.classes_))

    model.save("models/priority_ann.keras")
    joblib.dump(ann_prep, "models/priority_preprocessor.pkl")
    joblib.dump(le, "models/priority_encoder.pkl")

    print("\nAll models saved inside 'models/' successfully!")


if __name__ == "__main__":
    main()
