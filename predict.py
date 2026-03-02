import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ===============================
# PATH SETUP
# ===============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "saved_models", "facial_model.h5")

IMG_SIZE = 48

# ===============================
# LOAD MODEL
# ===============================

model = load_model(MODEL_PATH)
print("✅ Model Loaded Successfully")

emotion_labels = [
    'anger',
    'contempt',
    'disgust',
    'fear',
    'happy',
    'sadness',
    'surprise'
]

stress_emotions = ['anger', 'disgust', 'fear']

# ===============================
# FACE DETECTOR
# ===============================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Use CAP_DSHOW for Windows camera stability
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("📷 Camera Started. Press 'Q' to quit.")

# ===============================
# STRESS TRACKING
# ===============================

stress_scores = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face / 255.0
        face = face.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        predictions = model.predict(face, verbose=0)
        probabilities = predictions[0]

        emotion_index = np.argmax(probabilities)
        emotion = emotion_labels[emotion_index]

        # Calculate stress score
        stress_score = sum(
            probabilities[i]
            for i, label in enumerate(emotion_labels)
            if label in stress_emotions
        )

        stress_score = float(min(stress_score, 1.0))
        stress_scores.append(stress_score)

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Show emotion
        cv2.putText(frame, f"Emotion: {emotion}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        # Show stress
        cv2.putText(frame, f"Stress: {stress_score:.2f}",
                    (x, y + h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

    cv2.imshow("Facial Stress Detection", frame)

    # Exit when Q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===============================
# CLEANUP
# ===============================

cap.release()
cv2.destroyAllWindows()

# ===============================
# FINAL AVERAGE STRESS
# ===============================

if stress_scores:
    avg_stress = np.mean(stress_scores)
    print("\n📊 Final Average Stress Score:", round(avg_stress, 3))

    if avg_stress > 0.5:
        print("⚠️ Significant Stress Detected")
    else:
        print("✅ No Significant Stress Detected")
else:
    print("⚠️ No faces detected during session.")
