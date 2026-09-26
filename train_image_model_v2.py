import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train image model as a new version without overwriting current model.")
    parser.add_argument(
        "--output-model",
        default="",
        help="Optional output path (.keras). If omitted, uses models/image_model_YYYYMMDD_HHMMSS.keras",
    )
    return parser.parse_args()


# SETTINGS
IMG_SIZE = 224
BATCH_SIZE = 32
WARMUP_EPOCHS = 6
FINE_TUNE_EPOCHS = 8
SEED = 42


def main() -> None:
    args = parse_args()
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.output_model.strip():
        output_model_path = Path(args.output_model)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_model_path = models_dir / f"image_model_{stamp}.keras"

    best_model_path = output_model_path.with_name(output_model_path.stem + "_best.keras")

    # DATASET
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
        rotation_range=15,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.15,
        brightness_range=(0.8, 1.2),
        horizontal_flip=True,
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
    )

    train_data = train_datagen.flow_from_directory(
        "dataset",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=SEED,
    )

    val_data = val_datagen.flow_from_directory(
        "dataset",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=SEED,
    )

    # LOAD PRETRAINED MODEL
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )

    # Freeze pretrained layers
    base_model.trainable = False

    # BUILD MODEL
    model = Sequential(
        [
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.4),
            Dense(256, activation="relu"),
            Dropout(0.3),
            Dense(train_data.num_classes, activation="softmax"),
        ]
    )

    # Class weights help reduce bias from class imbalance.
    train_counts = np.bincount(train_data.classes, minlength=train_data.num_classes)
    mean_count = np.mean(train_counts)
    class_weights = {i: float(mean_count / c) for i, c in enumerate(train_counts)}

    print("Class indices:", train_data.class_indices)
    print("Train counts:", train_counts.tolist())
    print("Class weights:", class_weights)
    print("Saving new model to:", output_model_path)
    print("Saving best checkpoint to:", best_model_path)

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
        ),
        ModelCheckpoint(
            str(best_model_path),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    # WARMUP TRAINING
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )

    history_warmup = model.fit(
        train_data,
        validation_data=val_data,
        epochs=WARMUP_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # FINE-TUNE THE TOP OF MOBILENETV2
    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )

    history_finetune = model.fit(
        train_data,
        validation_data=val_data,
        initial_epoch=WARMUP_EPOCHS,
        epochs=WARMUP_EPOCHS + FINE_TUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # SAVE MODEL
    model.save(str(output_model_path))
    print("Model saved as", output_model_path)
    print("Best checkpoint:", best_model_path)

    # PLOT ACCURACY
    acc = history_warmup.history["accuracy"] + history_finetune.history["accuracy"]
    val_acc = history_warmup.history["val_accuracy"] + history_finetune.history["val_accuracy"]

    plt.plot(acc, label="Train")
    plt.plot(val_acc, label="Validation")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
