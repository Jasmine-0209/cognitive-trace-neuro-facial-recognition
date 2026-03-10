import os
import sys
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import mne
from tensorflow.keras.models import load_model

print("===================================")
print("   Cognitive Trace Fusion App      ")
print("===================================")

# =========================
# FACIAL STRESS CONFIG
# =========================
IMG_SIZE = 48
FACIAL_MODEL_PATH = "saved_models/facial_model.h5"
emotion_labels = ['anger','contempt','disgust','fear','happy','sadness','surprise']
stress_emotions = ['anger','disgust','fear']

# =========================
# LOAD FACIAL MODEL
# =========================
if not os.path.exists(FACIAL_MODEL_PATH):
    raise FileNotFoundError(f"Facial model not found: {FACIAL_MODEL_PATH}")
facial_model = load_model(FACIAL_MODEL_PATH, compile=False)
print("✅ Facial model loaded")

# =========================
# FACE DETECTOR
# =========================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# =========================
# EEG PROCESSING FUNCTION
# =========================
def process_eeg(file_path, threshold=2.0):
    if not os.path.exists(file_path):
        print("⚠️ EEG file not found")
        return None
    data = sio.loadmat(file_path)
    eeg = data['test_data']
    print("EEG Shape:", eeg.shape)

    trials, samples, channels = eeg.shape
    sfreq = 240
    ch_names = [f"EEG{i+1}" for i in range(channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

    eeg = np.transpose(eeg, (0,2,1))
    epochs = mne.EpochsArray(eeg, info)
    evoked = epochs.average()
    avg_signal = np.mean(evoked.data, axis=0)

    start = int(0.25*sfreq)
    end = int(0.50*sfreq)
    peak = np.max(avg_signal[start:end])
    peak_time = (start + np.argmax(avg_signal[start:end])) / sfreq * 1000

    print(f"\nP300 Peak: {round(float(peak),3)} µV")
    print(f"Peak Latency: {round(peak_time,2)} ms")
    result = "Recognition Detected ✅" if peak>threshold else "No Recognition ❌"
    print("EEG Result:", result)

    probability = min(95, round((peak/threshold)*70 + 30,2)) if peak>threshold else max(5, round((peak/threshold)*50,2))
    return avg_signal, probability, samples, sfreq

# =========================
# LIVE FACIAL STRESS FUNCTION
# =========================
def run_facial_stress(eeg_prob=None):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    stress_scores = []
    print("Starting live facial stress detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face,(IMG_SIZE,IMG_SIZE))/255.0
            face = face.reshape(1,IMG_SIZE,IMG_SIZE,1)

            pred = facial_model.predict(face, verbose=0)[0]
            emotion_index = np.argmax(pred)
            emotion = emotion_labels[emotion_index]

            stress = sum(pred[i] for i,label in enumerate(emotion_labels) if label in stress_emotions)
            stress = min(stress,1.0)
            stress_scores.append(stress)

            # Fuse with EEG if available
            if eeg_prob:
                fused = 0.7*stress + 0.3*(eeg_prob/100)
            else:
                fused = stress
            fused_percent = int(fused*100)

            # Draw overlays
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame,f"Emotion: {emotion}",(x,y-30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            cv2.putText(frame,f"Fused Stress: {fused_percent}%",(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        cv2.imshow("Cognitive Trace - EEG+Facial Fusion", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if stress_scores:
        avg_facial = np.mean(stress_scores)*100
        print(f"\nAverage Facial Stress: {avg_facial:.2f}%")
        if eeg_prob:
            fused_score = 0.7*avg_facial + 0.3*eeg_prob
            print(f"Final Fused Stress Score: {fused_score:.2f}%")
        else:
            print(f"Final Stress Score (Facial Only): {avg_facial:.2f}%")
    else:
        print("No faces detected")

# =========================
# MAIN MENU
# =========================
while True:
    print("\nMenu:")
    print("1. Analyze EEG + Live Facial Stress")
    print("2. Exit")
    choice = input("Enter your choice: ")

    if choice=="1":
        eeg_file = input("Enter EEG file name (Example: S1.mat): ")
        try:
            threshold = float(input("Enter Detection Threshold (Example: 2.0): "))
        except:
            threshold = 2.0

        avg_signal, eeg_prob, samples, sfreq = process_eeg(eeg_file, threshold)
        run_facial_stress(eeg_prob)

        # Show EEG ERP plot
        times = np.arange(samples)/sfreq*1000
        plt.figure()
        plt.plot(times, avg_signal)
        plt.axvline(x=300, linestyle='--')
        plt.title(f"{eeg_file} - ERP Waveform")
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude")
        plt.show()

    elif choice=="2":
        print("Exiting...")
        sys.exit()
    else:
        print("Invalid choice ❌ Try again.")