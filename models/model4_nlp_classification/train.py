#!/usr/bin/env python3
"""
Model 4: NLP Classification — Drug Review Effectiveness
=========================================================
Classify patient drug reviews into 3 effectiveness categories:
"Highly Effective", "Somewhat Effective", "Ineffective"

Uses TF-IDF + Logistic Regression.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score
)
from pipelines.data_pipeline import prepare_review_data

SAVED_MODEL_DIR = PROJECT_ROOT / "models" / "model4_nlp_classification" / "saved_model"


def train_model(X_train, y_train):
    """Train LinearSVC (typically best for text classification)."""
    # LinearSVC is faster and often better than LogisticRegression for text
    base_model = LinearSVC(
        C=0.5,
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )
    # Wrap in CalibratedClassifierCV to get probability estimates
    model = CalibratedClassifierCV(base_model, cv=3)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_val, y_val):
    """Evaluate NLP model."""
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)

    print("\n" + "=" * 60)
    print("MODEL 4 EVALUATION — NLP Drug Review Classification")
    print("=" * 60)
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))

    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="weighted")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1: {f1:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_val, y_pred)}")

    return {"accuracy": accuracy, "f1": f1}


def analyze_key_phrases(vectorizer, model):
    """Extract key phrases for each effectiveness class."""
    feature_names = vectorizer.get_feature_names_out()

    # Get the underlying model coefficients
    base_model = model
    if hasattr(model, "calibrated_classifiers_"):
        base_model = model.calibrated_classifiers_[0].estimator
    if not hasattr(base_model, "coef_"):
        print("\n--- Key phrase analysis skipped (no coefficients) ---")
        return

    classes = model.classes_ if hasattr(model, "classes_") else base_model.classes_

    print("\n--- Key Phrases by Effectiveness Class ---")
    for i, cls in enumerate(classes):
        coef = base_model.coef_[i]
        top_idx = np.argsort(coef)[-15:][::-1]
        bottom_idx = np.argsort(coef)[:15]
        top_phrases = [feature_names[j] for j in top_idx]
        neg_phrases = [feature_names[j] for j in bottom_idx]
        print(f"\n{cls}:")
        print(f"  Top indicators: {', '.join(top_phrases[:10])}")
        print(f"  Negative indicators: {', '.join(neg_phrases[:10])}")


def main():
    print("Loading and preprocessing drug review data...")
    texts_train, texts_val, y_train, y_val, df = prepare_review_data()

    print(f"Training: {len(texts_train)} reviews")
    print(f"Validation: {len(texts_val)} reviews")
    print(f"Class distribution (train):")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  {cls}: {cnt} ({cnt / len(y_train) * 100:.1f}%)")

    # Vectorize text
    print("\nVectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=3,
        max_df=0.95,
        strip_accents="unicode",
    )
    X_train = vectorizer.fit_transform(texts_train)
    X_val = vectorizer.transform(texts_val)
    print(f"TF-IDF features: {X_train.shape[1]}")

    # Train
    print("\nTraining Logistic Regression...")
    model = train_model(X_train, y_train)

    # Evaluate
    metrics = evaluate_model(model, X_val, y_val)

    # Key phrase analysis
    analyze_key_phrases(vectorizer, model)

    # Save
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, SAVED_MODEL_DIR / "model.joblib")
    joblib.dump(vectorizer, SAVED_MODEL_DIR / "vectorizer.joblib")
    joblib.dump(metrics, SAVED_MODEL_DIR / "metrics.joblib")
    print(f"\nModel saved to {SAVED_MODEL_DIR}")
    print("Training complete!")


if __name__ == "__main__":
    main()
