import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_path = r"C:\Users\priya\OneDrive\Desktop\python\AI_Powered_Sign_Language_Translator\sign_dataset"

# Image Data Generator (Data Augmentation)
datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Load Dataset
train_data = datagen.flow_from_directory(
    dataset_path,  # Update with the actual dataset path
    target_size=(64, 64),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(64, 64),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

#print(f"✅ Loaded {train_data.samples} training images from {len(train_data.class_indices)} classes.")
#print(f"✅ Loaded {val_data.samples} validation images from {len(val_data.class_indices)} classes.")
#print(f"Classes: {train_data.class_indices}")
#Get number of classes
num_classes = len(train_data.class_indices)
print("Classes:", train_data.class_indices)

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")  # Output layer
])

# Compile Model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train Model
model.fit(train_data, validation_data=val_data, epochs=20)

# Save Model
model.save("sign_language_model.keras")
print("✅ Model training complete! Saved as sign_language_model.keras")
