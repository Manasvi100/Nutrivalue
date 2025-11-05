import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
import os
import json

# Paths
train_dir = "dataset/train"
test_dir = "dataset/test"

# Image settings
img_size = (128, 128)
batch_size = 32

# Data augmentation (helps improve model accuracy)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load data
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# --- Build transfer learning model ---
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Freeze base model layers

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dense(train_data.num_classes, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(
    train_data,
    epochs=8,  # You can increase to 12–15 for better accuracy
    validation_data=test_data
)

# Save model
os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/food_model.h5")

# Save class labels
labels = list(train_data.class_indices.keys())
with open("dataset/labels.txt", "w") as f:
    for label in labels:
        f.write(f"{label}\n")

print("✅ Model trained and saved successfully!")
print("✅ Labels saved to dataset/labels.txt")
