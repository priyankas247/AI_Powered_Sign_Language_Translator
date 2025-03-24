import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load trained model
model = load_model("sign_language_model.h5")

# Compile the model (solves the warning issue)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define class labels (modify based on your dataset)
CLASS_LABELS = ["0","1","2","3","4","5","6","7","8","9","a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

# Load and preprocess an image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))  # Resize to match training input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array

# ✅ Set path to the folder containing test images
test_images_folder = r"C:/Users/priya/OneDrive/Desktop/python/AI_Powered_Sign_Language_Translator/sign_dataset/0/hand1_0_bot_seg_1_cropped"

# Loop through all images in the folder
for root, dirs, files in os.walk(test_images_folder):
    for file in files:
        if file.endswith((".png", ".jpg", ".jpeg")):  # Process only image files
            test_image_path = os.path.join(root, file)
            
            print(f"Testing image: {test_image_path}")
            img_array = preprocess_image(test_image_path)

            # Predict
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions)
            confidence = np.max(predictions)

            print(f"Predicted Sign: {CLASS_LABELS[predicted_class]} (Confidence: {confidence:.2f})")

            # Display the image
            img = cv2.imread(test_image_path)
            if img is not None:
                cv2.putText(img, f"{CLASS_LABELS[predicted_class]} ({confidence:.2f})", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Prediction", img)
                cv2.waitKey(1000)  # Display each image for 1 second
                cv2.destroyAllWindows()
            else:
                print(f"Error: Could not load {test_image_path} for display.")

print("Testing completed for all images.")
