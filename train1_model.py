import os

dataset_path = r"C:\Users\priya\OneDrive\Desktop\python\AI_Powered_Sign_Language_Translator\sign_dataset"

for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_path):
        num_images = len([img for img in os.listdir(class_path) if img.endswith((".jpg", ".png", ".jpeg"))])
        print(f"Class '{class_name}' has {num_images} images.")
