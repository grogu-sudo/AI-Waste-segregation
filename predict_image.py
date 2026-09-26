import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("models/best_image_model.keras")

# Your class names
classes = ["glass", "metal", "organic", "plastic"]

# Load image
img = cv2.imread("test.jpg")
if img is None:
    raise FileNotFoundError("Could not load test.jpg")

# Resize to match training size
img = cv2.resize(img, (224, 224))

# Convert OpenCV BGR image to RGB for MobileNetV2 pipeline
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Normalize image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

img = preprocess_input(img)

# Add batch dimension
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img)

# Get class index
class_index = np.argmax(prediction)

# Get confidence
confidence = np.max(prediction) * 100

# Final label
label = classes[class_index]

print(f"Prediction : {label}")
print(f"Confidence : {confidence:.2f}%")
