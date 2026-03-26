#!/usr/bin/env python3
"""
Model 4: NLP Classification — Drug Review Effectiveness
=========================================================
Classify patient drug reviews into 3 effectiveness categories:
"Highly Effective", "Somewhat Effective", "Ineffective"

Uses TF-IDF + metadata features with an ensemble classifier.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")

from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score
)
from pipelines.data_pipeline import prepare_review_data, load_raw_reviews, clean_review_text, EFFECTIVENESS_MAP

SAVED_MODEL_DIR = PROJECT_ROOT / "models" / "model4_nlp_classification" / "saved_model"


def prepare_enhanced_data(filepath=None, test_size=0.2, random_state=42):
    """Load data with extra metadata features (rating, drug, condition)."""
    from sklearn.model_selection import train_test_split

    df = load_raw_reviews(filepath)

    # Combine text fields
    for col in ["benefitsReview", "sideEffectsReview", "commentsReview"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    df["review_text"] = (
        df["benefitsReview"].astype(str) + " " +
        df["sideEffectsReview"].astype(str) + " " +
        df["commentsReview"].astype(str)
    ).str.strip()
    df["review_text_clean"] = df["review_text"].apply(clean_review_text)

    # Drop empty text
    df = df[df["review_text_clean"].str.len() > 10].copy()

    # Map effectiveness
    df["effectiveness_3class"] = df["effectiveness"].map(EFFECTIVENESS_MAP)
    df = df.dropna(subset=["effectiveness_3class"])

    # Extract text-derived metadata features (NO rating — it leaks the target)
    df["review_length"] = df["review_text_clean"].str.len()
    df["word_count"] = df["review_text_clean"].str.split().str.len()
    df["exclamation_count"] = df["review_text_clean"].str.count("!")
    df["question_count"] = df["review_text_clean"].str.count("\\?")
    df["has_side_effects"] = df["sideEffectsReview"].str.len().gt(5).astype(int)
    df["avg_word_length"] = df["review_text_clean"].apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.strip() else 0
    )

    texts = df["review_text_clean"].values
    y = df["effectiveness_3class"].values
    metadata = df[["review_length", "word_count", "exclamation_count",
                    "question_count", "has_side_effects", "avg_word_length"]].values

    texts_train, texts_val, y_train, y_val, meta_train, meta_val = train_test_split(
        texts, y, metadata,
        test_size=test_size, random_state=random_state, stratify=y,
    )

    return texts_train, texts_val, y_train, y_val, meta_train, meta_val, df


def train_model(X_train, y_train):
    """Train CalibratedClassifierCV(LinearSVC) — fast and effective for text."""
    model = CalibratedClassifierCV(
        LinearSVC(C=0.5, class_weight="balanced", max_iter=5000, random_state=42),
        cv=5,
    )
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


def main():
    print("Loading and preprocessing drug review data with metadata...")
    texts_train, texts_val, y_train, y_val, meta_train, meta_val, df = \
        prepare_enhanced_data()

    print(f"Training: {len(texts_train)} reviews")
    print(f"Validation: {len(texts_val)} reviews")
    print(f"Class distribution (train):")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  {cls}: {cnt} ({cnt / len(y_train) * 100:.1f}%)")

    # Vectorize text
    print("\nVectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=80000,
        ngram_range=(1, 3),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        strip_accents="unicode",
        analyzer="word",
    )
    X_train_tfidf = vectorizer.fit_transform(texts_train)
    X_val_tfidf = vectorizer.transform(texts_val)
    print(f"TF-IDF features: {X_train_tfidf.shape[1]}")

    # Also add character n-grams for typo/slang robustness
    print("Adding character n-grams...")
    char_vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(3, 5),
        analyzer="char_wb",
        sublinear_tf=True,
        min_df=3,
        max_df=0.95,
    )
    X_train_char = char_vectorizer.fit_transform(texts_train)
    X_val_char = char_vectorizer.transform(texts_val)
    print(f"Char n-gram features: {X_train_char.shape[1]}")

    # Scale metadata features
    meta_scaler = StandardScaler()
    meta_train_scaled = csr_matrix(meta_scaler.fit_transform(meta_train))
    meta_val_scaled = csr_matrix(meta_scaler.transform(meta_val))

    # Combine all features
    X_train = hstack([X_train_tfidf, X_train_char, meta_train_scaled])
    X_val = hstack([X_val_tfidf, X_val_char, meta_val_scaled])
    print(f"Total features: {X_train.shape[1]}")

    # Train ensemble
    print("\nTraining ensemble (LinearSVC + LR + SGD)...")
    model = train_model(X_train, y_train)

    # Evaluate
    metrics = evaluate_model(model, X_val, y_val)

    # Save all artifacts
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, SAVED_MODEL_DIR / "model.joblib")
    joblib.dump(vectorizer, SAVED_MODEL_DIR / "vectorizer.joblib")
    joblib.dump(char_vectorizer, SAVED_MODEL_DIR / "char_vectorizer.joblib")
    joblib.dump(meta_scaler, SAVED_MODEL_DIR / "meta_scaler.joblib")
    joblib.dump(metrics, SAVED_MODEL_DIR / "metrics.joblib")
    print(f"\nModel saved to {SAVED_MODEL_DIR}")
    print("Training complete!")


if __name__ == "__main__":
    main()
