#!/usr/bin/env python3
"""
Model 4: NLP Classification — Prediction Script
=================================================
Loads trained NLP ensemble and generates predictions on raw test data.
Output: test_data/model4_results.csv
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix
from pipelines.data_pipeline import (
    preprocess_reviews_for_prediction, clean_review_text, find_test_csv,
)

MODEL_DIR = PROJECT_ROOT / "models" / "model4_nlp_classification" / "saved_model"
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
OUTPUT_FILE = TEST_DATA_DIR / "model4_results.csv"


def main():
    # Load all artifacts
    model = joblib.load(MODEL_DIR / "model.joblib")
    vectorizer = joblib.load(MODEL_DIR / "vectorizer.joblib")
    char_vectorizer = joblib.load(MODEL_DIR / "char_vectorizer.joblib")
    meta_scaler = joblib.load(MODEL_DIR / "meta_scaler.joblib")

    # Find test data
    test_csv = find_test_csv(
        TEST_DATA_DIR,
        expected_columns=["benefitsReview"],
        name_hint="medication",
    )
    raw_df = pd.read_csv(test_csv)
    print(f"Loaded test data: {test_csv.name} ({len(raw_df)} rows)")

    # Preprocess text
    df = preprocess_reviews_for_prediction(raw_df)
    texts = df["review_text_clean"].values

    # Extract text-derived metadata
    df["review_length"] = df["review_text_clean"].str.len()
    df["word_count"] = df["review_text_clean"].str.split().str.len()
    df["exclamation_count"] = df["review_text_clean"].str.count("!")
    df["question_count"] = df["review_text_clean"].str.count("\\?")
    se_col = raw_df["sideEffectsReview"].fillna("") if "sideEffectsReview" in raw_df.columns else ""
    df["has_side_effects"] = pd.Series(se_col).str.len().gt(5).astype(int)
    df["avg_word_length"] = df["review_text_clean"].apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.strip() else 0
    )

    metadata = df[["review_length", "word_count", "exclamation_count",
                    "question_count", "has_side_effects", "avg_word_length"]].values

    # Vectorize
    X_tfidf = vectorizer.transform(texts)
    X_char = char_vectorizer.transform(texts)
    meta_scaled = csr_matrix(meta_scaler.transform(metadata))
    X = hstack([X_tfidf, X_char, meta_scaled])

    # Predict
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    confidence = np.max(y_proba, axis=1)

    # Build output
    if "Patient ID" in raw_df.columns:
        id_col = raw_df["Patient ID"].values
    elif "id" in raw_df.columns:
        id_col = raw_df["id"].values
    else:
        id_col = range(len(raw_df))

    results = pd.DataFrame({
        "id": id_col,
        "predicted_class": y_pred,
        "confidence": np.round(confidence, 4),
    })

    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)
    print(f"Predictions saved to {OUTPUT_FILE} ({len(results)} rows)")


if __name__ == "__main__":
    main()
