#!/usr/bin/env python3
"""
Model 2: Deep Learning — Prediction Script
============================================
Loads trained DNN (ONNX or Keras) and generates predictions on raw test data.
Output: test_data/model2_results.csv
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
from pipelines.data_pipeline import (
    clean_encounters, find_test_csv, preprocess_encounters_for_prediction,
)

MODEL_DIR = PROJECT_ROOT / "models" / "model2_deep_learning" / "saved_model"
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
OUTPUT_FILE = TEST_DATA_DIR / "model2_results.csv"


def load_model():
    """Load model - try ONNX first, fall back to Keras."""
    onnx_path = MODEL_DIR / "model.onnx"
    if onnx_path.exists():
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(onnx_path))
            return ("onnx", session)
        except ImportError:
            pass
    # Fallback to Keras
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_DIR / "model.keras")
    return ("keras", model)


def predict_with_model(model_tuple, X):
    """Run prediction with either ONNX or Keras model."""
    model_type, model = model_tuple
    X = np.array(X, dtype=np.float32)
    if model_type == "onnx":
        input_name = model.get_inputs()[0].name
        result = model.run(None, {input_name: X})
        return result[0].flatten()
    else:
        return model.predict(X, verbose=0).flatten()


def main():
    model_tuple = load_model()
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
    feature_cols = joblib.load(MODEL_DIR / "feature_cols.joblib")

    test_csv = find_test_csv(
        TEST_DATA_DIR,
        expected_columns=["encounter_id", "patient_nbr"],
        name_hint="encounter",
    )
    raw_df = pd.read_csv(test_csv)
    print(f"Loaded test data: {test_csv.name} ({len(raw_df)} rows)")

    X, df_clean = preprocess_encounters_for_prediction(raw_df, preprocessor, feature_cols)

    y_proba = predict_with_model(model_tuple, X)
    y_pred = (y_proba >= 0.5).astype(int)
    confidence = np.maximum(y_proba, 1 - y_proba)

    id_col = raw_df["encounter_id"] if "encounter_id" in raw_df.columns else range(len(raw_df))

    results = pd.DataFrame({
        "id": id_col.values if hasattr(id_col, 'values') else id_col,
        "prediction": y_pred,
        "probability": np.round(y_proba, 4),
        "confidence": np.round(confidence, 4),
    })

    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)
    print(f"Predictions saved to {OUTPUT_FILE} ({len(results)} rows)")


if __name__ == "__main__":
    main()
