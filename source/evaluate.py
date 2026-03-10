import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "saved_models", "facial_model.h5")
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset")

IMG_SIZE = 48
BATCH_SIZE = 32

emotion_labels = [
    'anger','contempt','disgust',
    'fear','happy','sadness','surprise'
]

stress_emotions = ['anger','disgust','fear']

model = load_model(MODEL_PATH)

images = []
total_images = 0

# ==============================
# LOAD ALL IMAGES FIRST
# ==============================

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
            face = face / 255.0
            face = face.reshape(IMG_SIZE, IMG_SIZE, 1)

            images.append(face)
            total_images += 1

# Convert to numpy array
images = np.array(images)

# ==============================
# BATCH PREDICTION (FAST)
# ==============================

predictions = model.predict(images, batch_size=BATCH_SIZE)

emotion_count = {emotion: 0 for emotion in emotion_labels}
stress_scores = []

for prediction in predictions:
    emotion_index = np.argmax(prediction)
    detected_emotion = emotion_labels[emotion_index]
    emotion_count[detected_emotion] += 1

    stress_score = sum(
        prediction[i] for i, label in enumerate(emotion_labels)
        if label in stress_emotions
    )
    stress_scores.append(stress_score)

# ==============================
# FINAL OUTPUT
# ==============================

avg_stress = np.mean(stress_scores)

if avg_stress >= 0.6:
    status = "HIGH STRESS DETECTED"
elif avg_stress >= 0.3:
    status = "MODERATE STRESS"
else:
    status = "NO SIGNIFICANT STRESS"

print("\n===== DATASET ANALYSIS =====")
print("Total Images :", total_images)
print("Average Stress :", round(avg_stress,2))
print("Final Decision :", status)
print("============================")
