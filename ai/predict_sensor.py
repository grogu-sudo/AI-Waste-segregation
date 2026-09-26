import joblib

model = joblib.load("models/sensor_model.pkl")
scaler = joblib.load("models/scaler.pkl")


def predict_waste(moisture, ir, metal):

    features = [[moisture, ir, metal]]

    scaled = scaler.transform(features)

    prediction = model.predict(scaled)[0]

    probabilities = model.predict_proba(scaled)[0]

    confidence = max(probabilities) * 100

    return prediction, confidence
