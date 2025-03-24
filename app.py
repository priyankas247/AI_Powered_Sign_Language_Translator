from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Lazy load the model to save memory
model = None

def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model("sign_language_model.h5")  # Update with your model path

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the Sign Language Prediction API!"})

@app.route("/predict", methods=["POST"])
def predict():
    load_model()  # Load only when needed
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    image = Image.open(file).resize((224, 224))  # Adjust size as needed
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return jsonify({"prediction": str(predicted_class), "confidence": confidence})

if __name__ == "__main__":
    app.run(debug=True)
