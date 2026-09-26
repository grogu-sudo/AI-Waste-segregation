import time

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

# Load model
model = load_model("models/image_model.keras")

# Classes in alphabetical order (same as flow_from_directory)
classes = [
    "glass",
    "metal",
    "organic",
    "plastic",
]


def get_center_roi(frame: np.ndarray, scale: float = 0.78) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Return a centered square ROI that always stays inside the frame."""
    h, w = frame.shape[:2]
    size = int(min(h, w) * scale)
    size = max(120, size)
    half = size // 2
    cx, cy = w // 2, h // 2

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)

    roi = frame[y1:y2, x1:x2]
    return roi, (x1, y1, x2, y2)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    # Training pipeline uses RGB images; convert camera BGR -> RGB.
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)


def predict_average(frames: list[np.ndarray]) -> np.ndarray:
    batch = np.concatenate([preprocess_frame(f) for f in frames], axis=0)
    preds = model.predict(batch, verbose=0)
    return preds.mean(axis=0)


# Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

display_text = "Place object in box, press SPACE"
last_prediction_time = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi, (x1, y1, x2, y2) = get_center_roi(frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if time.time() - last_prediction_time > 3:
        display_text = "Place object in box, press SPACE"

    cv2.putText(
        frame,
        display_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        "Q = quit | C = clear smoothing",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 220, 0),
        1,
    )

    cv2.imshow("AI Waste Classifier", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):
        burst_frames = [roi]
        for _ in range(4):
            ret_burst, burst = cap.read()
            if not ret_burst:
                continue
            burst_roi, _ = get_center_roi(burst)
            if burst_roi.size > 0:
                burst_frames.append(burst_roi)

        if burst_frames:
            probs = predict_average(burst_frames)
            class_index = int(np.argmax(probs))
            label = classes[class_index]
            confidence = float(probs[class_index] * 100)

            top2 = np.argsort(probs)[::-1][:2]
            margin = float(probs[top2[0]] - probs[top2[1]])

            # Keep uncertain gate, but less strict for real camera usage.
            if confidence < 45 or margin < 0.05:
                display_text = "Uncertain - center object and retry"
            else:
                display_text = f"{label} ({confidence:.1f}%)"

            last_prediction_time = time.time()
            print(
                f"Top1: {classes[top2[0]]} {probs[top2[0]] * 100:.1f}% | "
                f"Top2: {classes[top2[1]]} {probs[top2[1]] * 100:.1f}%"
            )

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
