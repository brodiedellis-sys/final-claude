#!/usr/bin/env python3
"""
Model 4: NLP Classification — Prediction Script
=================================================
Loads trained NLP model and generates predictions on raw test data.
Output: test_data/model4_results.csv
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
from pipelines.data_pipeline import (
    preprocess_reviews_for_prediction, clean_review_text, find_test_csv,
)

MODEL_DIR = PROJECT_ROOT / "models" / "model4_nlp_classification" / "saved_model"
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
OUTPUT_FILE = TEST_DATA_DIR / "model4_results.csv"


def main():
    # Load model and vectorizer
    model = joblib.load(MODEL_DIR / "model.joblib")
    vectorizer = joblib.load(MODEL_DIR / "vectorizer.joblib")

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

    # Vectorize
    X = vectorizer.transform(texts)

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
