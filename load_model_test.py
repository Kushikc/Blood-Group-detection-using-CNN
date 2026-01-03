# from tensorflow.keras.models import load_model

# # Load the trained model
# MODEL_SAVE_PATH = 'blood_group_detector_mobilenetv2.h5'
# model = load_model(MODEL_SAVE_PATH)

# # Print model structure
# model.summary()

import os

# Path to your A+ folder
folder_path = r"C:\Users\rishi\OneDrive\Desktop\MajorProject\dataset_blood_group\A+"

if not os.path.exists(folder_path):
    print("⚠️ Folder path does not exist!")
else:
    # Filter only BMP files
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(".bmp")]
    
    if not files:
        print("⚠️ No BMP images found in folder!")
    else:
        print("✅ Found BMP files:", files[:5])  # show first 5 files
        img_path = os.path.join(folder_path, files[0])  # pick first image
        print("Selected image:", img_path)

