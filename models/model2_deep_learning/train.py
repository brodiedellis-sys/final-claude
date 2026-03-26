#!/usr/bin/env python3
"""
Model 2: Deep Learning — Keras DNN Readmission Prediction
==========================================================
Binary classification using a deep neural network.
Compared against Model 1 (XGBoost).
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

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, f1_score
)
from pipelines.data_pipeline import prepare_encounter_data

SAVED_MODEL_DIR = PROJECT_ROOT / "models" / "model2_deep_learning" / "saved_model"


def build_model(input_dim):
    """Build a 4-layer DNN with dropout and batch normalization."""
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def train_model(model, X_train, y_train, X_val, y_val):
    """Train with early stopping and class weights."""
    # Class weights for imbalance
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    total = n_neg + n_pos
    class_weight = {
        0: total / (2 * n_neg),
        1: total / (2 * n_pos),
    }

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=10, mode="max",
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5,
            min_lr=1e-6, verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=256,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def evaluate_model(model, X_val, y_val):
    """Evaluate and print metrics."""
    y_proba = model.predict(X_val, verbose=0).flatten()
    y_pred = (y_proba >= 0.5).astype(int)

    print("\n" + "=" * 60)
    print("MODEL 2 EVALUATION — DNN Readmission Prediction")
    print("=" * 60)
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=["No Readmit", "Readmit"]))

    auc = roc_auc_score(y_val, y_proba)
    f1 = f1_score(y_val, y_pred, average="weighted")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Weighted F1: {f1:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_val, y_pred)}")

    return {"auc": auc, "f1": f1}


def plot_training_curves(history):
    """Save training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(history.history["loss"], label="Train")
    axes[0].plot(history.history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].legend()

    # Accuracy
    axes[1].plot(history.history["accuracy"], label="Train")
    axes[1].plot(history.history["val_accuracy"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    # AUC
    axes[2].plot(history.history["auc"], label="Train")
    axes[2].plot(history.history["val_auc"], label="Val")
    axes[2].set_title("AUC")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(SAVED_MODEL_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Training curves saved.")


def main():
    print("Loading and preprocessing encounter data...")
    X_train, X_val, y_train, y_val, preprocessor, feature_cols, df = \
        prepare_encounter_data()

    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Validation set: {X_val.shape[0]} samples")

    # Build model
    model = build_model(input_dim=X_train.shape[1])
    model.summary()

    # Train
    print("\nTraining DNN model...")
    history = train_model(model, X_train, y_train, X_val, y_val)

    # Evaluate
    metrics = evaluate_model(model, X_val, y_val)

    # Plot training curves
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    plot_training_curves(history)

    # Compare with Model 1 if available
    m1_metrics_path = PROJECT_ROOT / "models" / "model1_traditional_ml" / "saved_model" / "metrics.joblib"
    if m1_metrics_path.exists():
        m1_metrics = joblib.load(m1_metrics_path)
        print("\n" + "=" * 60)
        print("MODEL COMPARISON: Model 1 (XGBoost) vs Model 2 (DNN)")
        print("=" * 60)
        print(f"  {'Metric':<20} {'XGBoost':>10} {'DNN':>10}")
        print(f"  {'AUC-ROC':<20} {m1_metrics['auc']:>10.4f} {metrics['auc']:>10.4f}")
        print(f"  {'Weighted F1':<20} {m1_metrics['f1']:>10.4f} {metrics['f1']:>10.4f}")

    # Save model
    model.save(SAVED_MODEL_DIR / "model.keras")
    joblib.dump(preprocessor, SAVED_MODEL_DIR / "preprocessor.joblib")
    joblib.dump(feature_cols, SAVED_MODEL_DIR / "feature_cols.joblib")
    joblib.dump(metrics, SAVED_MODEL_DIR / "metrics.joblib")
    print(f"\nModel saved to {SAVED_MODEL_DIR}")
    print("Training complete!")


if __name__ == "__main__":
    main()
