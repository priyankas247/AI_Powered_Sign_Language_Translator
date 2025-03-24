import cv2
import os
import numpy as np
import pickle

IMG_SIZE = 64  # Resize images to 64x64
DATASET_PATH = "sign_dataset"
OUTPUT_PATH = "preprocessed_data.pkl"  # Save processed data for training

def load_and_preprocess_data():
    X, y = [], []
    class_labels = os.listdir(DATASET_PATH)
    class_labels.sort()  # Ensure consistent order of labels

    for idx, label in enumerate(class_labels):
        label_path = os.path.join(DATASET_PATH, label)
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            
            # Read and resize image
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0  # Normalize pixel values (0 to 1)
            
            X.append(img)
            y.append(idx)

    X = np.array(X)
    y = np.array(y)

    # Save processed data using pickle
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump((X, y, class_labels), f)
    
    print(f"✅ Preprocessing complete! Data saved to {OUTPUT_PATH}")
    return X, y, class_labels

if __name__ == "__main__":
    load_and_preprocess_data()
