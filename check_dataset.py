import os

data_path = "dataset_blood_group/"

# Check available blood group classes
print("Classes found:", os.listdir(data_path))

# Example: Count how many images are in A+ class
class_name = "A+"
print(f"Number of images in {class_name}: {len(os.listdir(os.path.join(data_path, class_name)))}")
