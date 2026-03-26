#!/usr/bin/env python3
"""
Model 1: Traditional ML — XGBoost Readmission Prediction
=========================================================
Binary classification: will this patient be readmitted?
Uses XGBoost with comprehensive feature engineering and SHAP analysis.
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
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, f1_score
)
from pipelines.data_pipeline import prepare_encounter_data

SAVED_MODEL_DIR = PROJECT_ROOT / "models" / "model1_traditional_ml" / "saved_model"


def train_model(X_train, y_train, X_val, y_val):
    """Train XGBoost with tuned hyperparameters."""
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos = n_neg / max(n_pos, 1)

    model = XGBClassifier(
        n_estimators=1500,
        max_depth=5,
        learning_rate=0.02,
        subsample=0.75,
        colsample_bytree=0.75,
        min_child_weight=8,
        gamma=0.15,
        reg_alpha=0.05,
        reg_lambda=1.2,
        scale_pos_weight=scale_pos,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=80,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )
    return model


def evaluate_model(model, X_val, y_val):
    """Evaluate and print metrics."""
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    print("\n" + "=" * 60)
    print("MODEL 1 EVALUATION — XGBoost Readmission Prediction")
    print("=" * 60)
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=["No Readmit", "Readmit"]))

    auc = roc_auc_score(y_val, y_proba)
    f1 = f1_score(y_val, y_pred, average="weighted")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Weighted F1: {f1:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_val, y_pred)}")

    return {"auc": auc, "f1": f1}


def explain_model(model, X_val, feature_names=None):
    """SHAP feature importance analysis."""
    try:
        import shap
        print("\nGenerating SHAP analysis...")
        explainer = shap.TreeExplainer(model)

        sample_size = min(1000, X_val.shape[0])
        X_sample = X_val[:sample_size]
        shap_values = explainer.shap_values(X_sample)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, X_sample,
            feature_names=feature_names,
            show=False, max_display=20,
        )
        plt.tight_layout()
        plt.savefig(SAVED_MODEL_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, X_sample,
            feature_names=feature_names,
            plot_type="bar",
            show=False, max_display=20,
        )
        plt.tight_layout()
        plt.savefig(SAVED_MODEL_DIR / "shap_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("SHAP plots saved.")

    except Exception as e:
        print(f"SHAP analysis note: {e}")
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            if feature_names and len(feature_names) == len(importances):
                indices = np.argsort(importances)[-20:]
                plt.figure(figsize=(10, 8))
                plt.barh(range(len(indices)), importances[indices])
                plt.yticks(range(len(indices)),
                           [feature_names[i] for i in indices])
                plt.title("Top 20 Feature Importances")
                plt.tight_layout()
                plt.savefig(SAVED_MODEL_DIR / "feature_importance.png",
                            dpi=150, bbox_inches="tight")
                plt.close()


def main():
    print("Loading and preprocessing encounter data...")
    X_train, X_val, y_train, y_val, preprocessor, feature_cols, df = \
        prepare_encounter_data()

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Feature dimensions: {X_train.shape[1]}")
    print(f"Positive class ratio: {y_train.mean():.3f}")

    # Train
    print("\nTraining XGBoost model...")
    model = train_model(X_train, y_train, X_val, y_val)

    # Evaluate
    metrics = evaluate_model(model, X_val, y_val)

    # Get feature names
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = None

    # Explain
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    explain_model(model, X_val, feature_names)

    # Save
    joblib.dump(model, SAVED_MODEL_DIR / "model.joblib")
    joblib.dump(preprocessor, SAVED_MODEL_DIR / "preprocessor.joblib")
    joblib.dump(feature_cols, SAVED_MODEL_DIR / "feature_cols.joblib")
    joblib.dump(metrics, SAVED_MODEL_DIR / "metrics.joblib")
    print(f"\nModel saved to {SAVED_MODEL_DIR}")
    print("Training complete!")


if __name__ == "__main__":
    main()
