import os
import random
import shutil

# Paths
data_dir = "dataset/images"
output_dir = "dataset"
train_ratio = 0.8  # 80% train, 20% test

# Create output folders
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Loop through each food class
for food_class in os.listdir(data_dir):
    class_path = os.path.join(data_dir, food_class)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    split_index = int(len(images) * train_ratio)
    train_images = images[:split_index]
    test_images = images[split_index:]

    os.makedirs(os.path.join(train_dir, food_class), exist_ok=True)
    os.makedirs(os.path.join(test_dir, food_class), exist_ok=True)

    for img in train_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(train_dir, food_class, img)
        )

    for img in test_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(test_dir, food_class, img)
        )

print("✅ Dataset successfully split into 'train/' and 'test/' folders.")
