#!/usr/bin/env python3
"""
Model 3: CNN — Diabetic Retinopathy Detection
===============================================
Binary classification: No DR (0) vs Has DR (1)
Uses a custom CNN with data augmentation.
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
    classification_report, confusion_matrix, f1_score
)
from PIL import Image
from pipelines.data_pipeline import prepare_retinal_data

SAVED_MODEL_DIR = PROJECT_ROOT / "models" / "model3_cnn" / "saved_model"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def load_images_from_df(df, target_size=IMG_SIZE):
    """Load and preprocess images from a DataFrame with image_path and binary_label."""
    images = []
    labels = []
    for _, row in df.iterrows():
        try:
            img = Image.open(row["image_path"]).convert("RGB")
            img = img.resize(target_size)
            img_array = np.array(img, dtype=np.float32) / 255.0
            images.append(img_array)
            labels.append(row["binary_label"])
        except Exception as e:
            print(f"Warning: Could not load {row['image_path']}: {e}")
            continue
    return np.array(images), np.array(labels)


def create_augmented_dataset(X, y, batch_size=BATCH_SIZE):
    """Create a tf.data.Dataset with augmentation."""
    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomContrast(0.1),
    ])

    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(len(X))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(
        lambda x, y_: (data_augmentation(x, training=True), y_),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def build_model():
    """Build CNN with transfer learning using MobileNetV2."""
    base_model = keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    # Freeze base model initially
    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc"),
                 keras.metrics.Recall(name="sensitivity")],
    )
    return model, base_model


def fine_tune_model(model, base_model):
    """Unfreeze top layers of base model for fine-tuning."""
    base_model.trainable = True
    # Freeze all but last 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc"),
                 keras.metrics.Recall(name="sensitivity")],
    )
    return model


def train_model(model, base_model, X_train, y_train, X_val, y_val):
    """Two-phase training: frozen base, then fine-tune."""
    # Class weights
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    total = n_neg + n_pos
    class_weight = {0: total / (2 * n_neg), 1: total / (2 * n_pos)}

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=8, mode="max",
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-7, verbose=1,
        ),
    ]

    # Create augmented training dataset
    train_ds = create_augmented_dataset(X_train, y_train)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)

    # Phase 1: Train classifier head
    print("\n--- Phase 1: Training classifier head ---")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # Phase 2: Fine-tune top layers
    print("\n--- Phase 2: Fine-tuning top layers ---")
    model = fine_tune_model(model, base_model)
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    return model, history1, history2


def evaluate_model(model, X_val, y_val):
    """Evaluate CNN performance."""
    y_proba = model.predict(X_val, verbose=0).flatten()
    y_pred = (y_proba >= 0.5).astype(int)

    print("\n" + "=" * 60)
    print("MODEL 3 EVALUATION — CNN Retinopathy Detection")
    print("=" * 60)
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=["No DR", "Has DR"]))

    accuracy = np.mean(y_pred == y_val)
    sensitivity = np.sum((y_pred == 1) & (y_val == 1)) / max(np.sum(y_val == 1), 1)
    specificity = np.sum((y_pred == 0) & (y_val == 0)) / max(np.sum(y_val == 0), 1)
    f1 = f1_score(y_val, y_pred, average="weighted")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity (DR detection): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Weighted F1: {f1:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_val, y_pred)}")

    return {"accuracy": accuracy, "sensitivity": sensitivity, "f1": f1}


def main():
    print("Loading retinal scan data...")
    train_df, val_df = prepare_retinal_data()
    print(f"Training: {len(train_df)} images, Validation: {len(val_df)} images")
    print(f"Train class distribution:\n{train_df['binary_label'].value_counts()}")

    print("\nLoading and preprocessing images...")
    X_train, y_train = load_images_from_df(train_df)
    X_val, y_val = load_images_from_df(val_df)
    print(f"Training images shape: {X_train.shape}")
    print(f"Validation images shape: {X_val.shape}")

    # Build model
    model, base_model = build_model()
    model.summary()

    # Train
    model, history1, history2 = train_model(
        model, base_model, X_train, y_train, X_val, y_val
    )

    # Evaluate
    metrics = evaluate_model(model, X_val, y_val)

    # Save
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(SAVED_MODEL_DIR / "model.keras")
    joblib.dump(metrics, SAVED_MODEL_DIR / "metrics.joblib")
    print(f"\nModel saved to {SAVED_MODEL_DIR}")
    print("Training complete!")


if __name__ == "__main__":
    main()
