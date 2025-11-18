NutriValue – AI-Based Indian Food Recognition & Calorie Estimation System
Authors

Manasvi Dhengre – manasvi.dhengre@cumminscollege.in

Kalyani Fulpagare – kalyani.fulpagare@cumminscollege.in

Mukta Chaudhari – mukta.chaudhari@cumminscollege.in

Department of Computer Engineering
MKSSS’s Cummins College of Engineering for Women, Pune, India

📌 Abstract

Accurate nutritional tracking is a major challenge for individuals trying to maintain a balanced diet. Manual calorie logging is time-consuming, inconsistent, and prone to human error. NutriValue is a web-based AI system that automatically identifies Indian food items from images and estimates calories and macronutrients using a fine-tuned MobileNetV2 CNN model.

The system integrates:

Deep learning–based food recognition

Flask web framework

MySQL database

Interactive analytics and gamification

Users can access daily and weekly nutrition breakdowns, receive real-time predictions, and engage with leaderboard features for motivation.


🧩 Problem Statement

Manual calorie tracking is tedious and error-prone.

Users lack awareness of nutritional content in everyday meals.

Indian cuisine is visually complex due to diverse textures and similar-looking dishes.

Goal:
Automate food recognition, provide accurate calorie/nutrient estimation, and encourage healthy habits through interactive tracking.


🎯 Objectives

Identify food items using MobileNetV2 CNN.

Estimate calories, protein, carbohydrates, and fats.

Log meals and generate daily/weekly nutrition analytics.

Provide personalized health tips.

Improve user engagement using streaks and leaderboards.



✨ Key Features
✔ Food Image Classification

Predicts 16 Indian food items using a fine-tuned MobileNetV2 model.

✔ Nutrition Estimation

Retrieves calories, protein, carbs, and fats from a structured JSON nutrition database.

✔ Daily & Weekly Tracker

Logs meals and visualizes calorie + macro trends using charts.

✔ Leaderboard

Ranks users based on total calorie logging activity and streaks.

✔ Health Tips

Provides personalized suggestions generated based on meal type.


🛠 System Architecture
🔄 Workflow:
User uploads image 
→ Preprocessing 
→ MobileNet CNN predicts food class 
→ JSON database fetches nutrition 
→ Tracker & leaderboard updated 
→ Results displayed with charts


📊 Model Training & Evaluation
Dataset Split

Training: 80%

Testing: 20%

Performance Metrics

| Metric         | Value |
| -------------- | ----- |
| Top-1 Accuracy | 92%   |
| Precision      | 92%   |
| Recall         | 90%   |
| F1 Score       | 91%   |


📷 Performance Under Various Conditions 

| Condition         | Accuracy |
| ----------------- | -------- |
| Normal lighting   | 92%      |
| Low lighting      | 87%      |
| Side-angle images | 89%      |


🏁 Conclusion

NutriValue provides an accurate, automated, and user-friendly platform for identifying Indian food and generating calorie estimates. 
With deep learning, a tracking dashboard, and user engagement modules, the system encourages healthier eating habits and simplifies nutritional awareness.
