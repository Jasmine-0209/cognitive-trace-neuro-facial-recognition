import os
import numpy as np
import scipy.io as sio
import mne
import cv2
from tensorflow.keras.models import load_model

# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EEG_DATASET_PATH = os.path.join(BASE_DIR, "..", "Dataset", "EEG Dataset")
FACIAL_DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "..", "saved_models", "facial_model.h5")

SFREQ = 240
IMG_SIZE = 48
BATCH_SIZE = 32

emotion_labels = [
    'anger','contempt','disgust',
    'fear','happy','sadness','surprise'
]

stress_emotions = ['anger','disgust','fear']

# =========================================================
# EEG MODULE
# =========================================================

def eeg_analysis(file_name, threshold):

    file_path = os.path.join(EEG_DATASET_PATH, file_name)

    if not os.path.exists(file_path):
        print("❌ EEG file not found.")
        return None, None

    data = sio.loadmat(file_path)
    eeg = data['test_data']

    trials, samples, channels = eeg.shape
    eeg = np.transpose(eeg, (0, 2, 1))

    ch_names = [f"EEG{i+1}" for i in range(channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types="eeg")

    epochs = mne.EpochsArray(eeg, info, verbose=False)

    # Safe filtering (fix for short signals)
    epochs.filter(l_freq=0.5, h_freq=20, method='iir', verbose=False)

    # Baseline correction
    epochs.apply_baseline((0, 0.2))

    evoked = epochs.average()
    avg_signal = np.mean(evoked.data, axis=0)

    # P300 window 250–500 ms
    start = int(0.25 * SFREQ)
    end = int(0.50 * SFREQ)

    window = avg_signal[start:end]
    peak = np.max(np.abs(window))

    print("\n📊 ===== EEG DEBUG INFO =====")
    print("P300 Time Window: 250–500 ms")
    print("Actual P300 Peak:", round(peak, 4))
    print("User Threshold:", threshold)
    print("=============================")

    # Stable probability formula
    probability = round((peak / (peak + threshold)) * 100, 2)

    print("🧠 EEG Recognition Probability:", probability, "%")

    return probability, peak


# =========================================================
# FACIAL MODULE
# =========================================================

def facial_analysis():

    if not os.path.exists(MODEL_PATH):
        print("❌ Facial model not found.")
        return None

    model = load_model(MODEL_PATH)

    images = []

    for root, dirs, files in os.walk(FACIAL_DATASET_PATH):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):

                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)

                if img is None:
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
                face = face / 255.0
                face = face.reshape(IMG_SIZE, IMG_SIZE, 1)

                images.append(face)

    if len(images) == 0:
        print("❌ No facial images found.")
        return None

    images = np.array(images)

    predictions = model.predict(images, batch_size=BATCH_SIZE, verbose=0)

    stress_scores = []

    for prediction in predictions:
        stress_score = sum(
            prediction[i] for i, label in enumerate(emotion_labels)
            if label in stress_emotions
        )
        stress_scores.append(stress_score)

    avg_stress = np.mean(stress_scores)
    probability = round(avg_stress * 100, 2)

    print("😐 Facial Stress Probability:", probability, "%")

    return probability


# =========================================================
# FUSION MODULE WITH EXPLANATION
# =========================================================

def fusion(eeg_prob, facial_prob, threshold, peak, w1=0.7, w2=0.3):

    final_score = round((w1 * eeg_prob) + (w2 * facial_prob), 2)

    if final_score >= 70:
        decision = "⚠️ High Probability of Concealed Recognition"
    elif final_score >= 40:
        decision = "Moderate Probability"
    else:
        decision = "Low Probability"

    # Explanation
    reason = ""

    if peak > threshold:
        reason += "• P300 peak exceeded threshold → strong neural recognition detected.\n"
    else:
        reason += "• P300 peak below threshold → weak neural recognition.\n"

    if eeg_prob > facial_prob:
        reason += "• EEG contributed more to final confidence (higher weight 70%).\n"
    else:
        reason += "• Facial stress contributed significantly to final score.\n"

    if facial_prob > 50:
        reason += "• High stress-related facial emotions detected.\n"
    else:
        reason += "• No strong stress indicators in facial analysis.\n"

    print("\n========== FINAL RESULT ==========")
    print("EEG (70%) :", eeg_prob, "%")
    print("Facial (30%) :", facial_prob, "%")
    print("Final Confidence :", final_score, "%")
    print("Decision :", decision)
    print("\n🧠 Explanation:")
    print(reason)
    print("==================================")

    return final_score


# =========================================================
# MAIN SYSTEM
# =========================================================

def main():

    print("========================================")
    print("   Cognitive Trace - Integrated System  ")
    print("========================================")

    file_name = input("Enter EEG file name (Example: S1.mat): ")

    try:
        threshold = float(input("Enter Detection Threshold (Example: 50): "))
    except:
        print("❌ Invalid threshold.")
        return

    eeg_prob, peak = eeg_analysis(file_name, threshold)

    if eeg_prob is None:
        return

    facial_prob = facial_analysis()

    if facial_prob is None:
        return

    fusion(eeg_prob, facial_prob, threshold, peak)


if __name__ == "__main__":
    main()
