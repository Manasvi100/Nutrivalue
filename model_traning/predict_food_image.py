import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json
import os

# Load model
model = tf.keras.models.load_model("saved_model/food_model.h5")

# Load class names
train_dir = "dataset/train"
class_names = sorted(os.listdir(train_dir))

# Optional calorie info (you can expand this)
calories = {
    "biryani": 290,
    "dosa": 133,
    "samosa": 262,
    "idli": 58,
    "pulao": 250,
    "butter_chicken": 490,
    "poha": 180
}

# Optional health tips (expand later!)
health_tips = {
    "biryani": "Try using brown rice and less oil for a healthier version.",
    "dosa": "Pair with chutney instead of oily sambar for fewer calories.",
    "samosa": "Air fry instead of deep frying to reduce fat.",
    "idli": "Great low-calorie breakfast option, rich in carbs and easy to digest.",
    "pulao": "Add more veggies and less ghee for better nutrition.",
    "butter_chicken": "Use grilled chicken and low-fat yogurt for a lighter recipe.",
    "poha": "Add peanuts for protein but control oil quantity."
}

# --- Predict function ---
def predict_food(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(100 * np.max(prediction), 2)

    cal = calories.get(predicted_class, "N/A")
    tip = health_tips.get(predicted_class, "Eat in moderation and stay active!")

    print(f"🍽️ Predicted Food: {predicted_class}")
    print(f"🔥 Confidence: {confidence}%")
    print(f"🥗 Estimated Calories: {cal} kcal")
    print(f"💡 Health Tip: {tip}")

# --- Example Usage ---
# Replace this path with any image from your test folder
img_path = "model_training/dataset/test/biryani/0d81432b55.jpg"
predict_food(img_path)
