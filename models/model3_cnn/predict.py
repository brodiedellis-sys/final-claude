#!/usr/bin/env python3
"""
Model 3: CNN — Prediction Script
==================================
Loads trained CNN and generates predictions on test retinal images.
Output: test_data/model3_results.csv
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from pipelines.data_pipeline import find_test_images

MODEL_DIR = PROJECT_ROOT / "models" / "model3_cnn" / "saved_model"
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
OUTPUT_FILE = TEST_DATA_DIR / "model3_results.csv"
IMG_SIZE = (224, 224)


def load_and_preprocess_images(image_dir):
    """Load all PNG images from directory."""
    image_dir = Path(image_dir)
    images = []
    image_ids = []

    for img_path in sorted(image_dir.glob("*.png")):
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(IMG_SIZE)
            img_array = np.array(img, dtype=np.float32) / 255.0
            images.append(img_array)
            image_ids.append(img_path.name)
        except Exception as e:
            print(f"Warning: Could not load {img_path.name}: {e}")
            continue

    return np.array(images), image_ids


def main():
    # Load model
    model = tf.keras.models.load_model(MODEL_DIR / "model.keras")

    # Find test images
    try:
        image_dir = find_test_images(TEST_DATA_DIR)
    except FileNotFoundError:
        # Fallback: use raw data images for testing
        image_dir = PROJECT_ROOT / "data" / "raw" / "retinal_scan_images"
        if not image_dir.exists():
            print("ERROR: No test images found")
            return

    print(f"Loading images from {image_dir}")
    images, image_ids = load_and_preprocess_images(image_dir)
    print(f"Loaded {len(images)} images")

    if len(images) == 0:
        print("ERROR: No images loaded")
        return

    # Predict
    y_proba = model.predict(images, verbose=0).flatten()
    y_pred = (y_proba >= 0.5).astype(int)
    confidence = np.maximum(y_proba, 1 - y_proba)

    results = pd.DataFrame({
        "image_id": image_ids,
        "predicted_class": y_pred,
        "confidence": np.round(confidence, 4),
    })

    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)
    print(f"Predictions saved to {OUTPUT_FILE} ({len(results)} rows)")


if __name__ == "__main__":
    main()
