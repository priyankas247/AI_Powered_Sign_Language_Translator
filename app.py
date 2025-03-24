from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# Load trained model
model = tf.keras.models.load_model("sign_language_model.h5")

# Recompile the model to fix the warning
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Class labels
CLASS_LABELS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", 
                "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", 
                "u", "v", "w", "x", "y", "z"]

app = Flask(__name__)

# ✅ Fix 404 Error: Add a home route
@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Sign Language Prediction API!"})

def preprocess_image(img_data):
    img = Image.open(io.BytesIO(img_data))  # Open image from bytes
    img = img.resize((64, 64))  # Resize to match model input size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    img_data = file.read()  # Read file as bytes

    # Preprocess image
    img_array = preprocess_image(img_data)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = float(np.max(predictions))

    return jsonify({"prediction": CLASS_LABELS[predicted_class], "confidence": confidence})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)  # Change port if needed

