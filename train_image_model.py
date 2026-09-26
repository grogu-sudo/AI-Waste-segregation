import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import matplotlib.pyplot as plt

# SETTINGS
IMG_SIZE = 224
BATCH_SIZE = 32
WARMUP_EPOCHS = 6
FINE_TUNE_EPOCHS = 8
SEED = 42

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
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    "dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=SEED
)

val_data = val_datagen.flow_from_directory(
    "dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=SEED
)

# LOAD PRETRAINED MODEL
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze pretrained layers
base_model.trainable = False

# BUILD MODEL
model = Sequential([
    base_model,

    GlobalAveragePooling2D(),

    Dropout(0.4),

    Dense(256, activation="relu"),

    Dropout(0.3),

    Dense(
        train_data.num_classes,
        activation="softmax"
    )
])

# Class weights help reduce bias from class imbalance.
train_counts = np.bincount(train_data.classes, minlength=train_data.num_classes)
mean_count = np.mean(train_counts)
class_weights = {i: float(mean_count / c) for i, c in enumerate(train_counts)}

print("Class indices:", train_data.class_indices)
print("Train counts:", train_counts.tolist())
print("Class weights:", class_weights)

callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-6
    ),
    ModelCheckpoint(
        "models/best_image_model.keras",
        monitor="val_accuracy",
        save_best_only=True
    )
]

# WARMUP TRAINING
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"]
)

history_warmup = model.fit(
    train_data,
    validation_data=val_data,
    epochs=WARMUP_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# FINE-TUNE THE TOP OF MOBILENETV2
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"]
)

history_finetune = model.fit(
    train_data,
    validation_data=val_data,
    initial_epoch=WARMUP_EPOCHS,
    epochs=WARMUP_EPOCHS + FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# SAVE MODEL
model.save("models/image_model.keras")

print("Model saved as models/image_model.keras")
print("Best checkpoint: models/best_image_model.keras")

# Merge history for plotting
acc = history_warmup.history["accuracy"] + history_finetune.history["accuracy"]
val_acc = history_warmup.history["val_accuracy"] + history_finetune.history["val_accuracy"]

# PLOT ACCURACY
plt.plot(acc, label="Train")
plt.plot(val_acc, label="Validation")

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

plt.show()
