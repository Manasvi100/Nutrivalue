from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image
import json

app = Flask(__name__)

# --- Load trained model ---
model = tf.keras.models.load_model('saved_model/food_model.h5')

# --- Load labels ---
with open('dataset/labels.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# --- Load calorie data ---
calorie_file = os.path.join(os.path.dirname(__file__), 'data', 'calorie.json')
if os.path.exists(calorie_file):
    with open(calorie_file, 'r') as f:
        calorie_data = json.load(f)
else:
    calorie_data = {}
print("Loaded calorie data keys:", calorie_data.keys())

# --- Optional health tips ---
health_tips = {
    "biryani": "Try using brown rice and less oil for a healthier version.",
    "dosa": "Pair with chutney instead of oily sambar for fewer calories.",
    "samosa": "Air fry instead of deep frying to reduce fat.",
    "idli": "Light and healthy breakfast option, rich in carbs and easy to digest.",
    "pulao": "Add more veggies and less ghee for better nutrition.",
    "butter chicken": "Use grilled chicken and low-fat yogurt for a lighter recipe.",
    "poha": "Add peanuts for protein but control oil quantity.",
    "besan laddu": "Use jaggery instead of sugar and ghee in moderation.",
    "chole bhature": "Limit portion size — high in calories but delicious!",
    "vada pav": "Enjoy occasionally; opt for air-fried version to cut fat."
}

# --- Ensure upload folder exists ---
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# --- Home route ---
@app.route('/')
def home():
    return render_template('index.html')


# --- Prediction route ---
@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Preprocess
    img = Image.open(filepath).resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_label = class_names[np.argmax(score)]
    confidence = round(100 * np.max(score), 2)

    # Fetch calorie + health tip (case-insensitive)
    label_key = predicted_label.lower().replace(" ", "_")
    calories = calorie_data.get(label_key, "Calorie data not available")
    tip = health_tips.get(label_key, "Eat in moderation and stay active!")

    return render_template(
        'index.html',
        prediction=predicted_label,
        confidence=confidence,
        calories=calories,
        tip=tip,
        img_path=filepath
    )

if __name__ == '__main__':
    app.run(debug=True)
