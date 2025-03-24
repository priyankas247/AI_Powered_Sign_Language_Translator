import cv2
import os

# Define the number of images per sign and the output directory
SIGN_CLASSES = ["Hello", "Thank_You", "Yes", "No", "Stop", "Peace", "I_Love_You"]  # Modify as needed
IMAGES_PER_SIGN = 1  # Capture 200 images per sign
SAVE_PATH = "sign_dataset"  # Folder where images will be saved

# Create directories for each class
for sign in SIGN_CLASSES:
    os.makedirs(os.path.join(SAVE_PATH, sign), exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)
current_sign = 0
image_count = 0

while current_sign < len(SIGN_CLASSES):
    ret, frame = cap.read()
    if not ret:
        break

    # Show current sign to display
    cv2.putText(frame, f"Show Sign: {SIGN_CLASSES[current_sign]}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Collecting Sign Data", frame)

    key = cv2.waitKey(1)

    # Press 'c' to capture an image
    if key & 0xFF == ord('c'):
        image_path = os.path.join(SAVE_PATH, SIGN_CLASSES[current_sign], f"{image_count}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Captured {image_count + 1}/{IMAGES_PER_SIGN} for {SIGN_CLASSES[current_sign]}")
        image_count += 1

        # Move to next sign after collecting enough images
        if image_count >= IMAGES_PER_SIGN:
            current_sign += 1
            image_count = 0

    # Press 'q' to exit
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
