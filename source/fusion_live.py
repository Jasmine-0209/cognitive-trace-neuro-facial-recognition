import os
import sys
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import mne
from tensorflow.keras.models import load_model

print("===================================")
print("   Cognitive Trace Real-Time Fusion ")
print("===================================")

# =========================
# CONFIG
# =========================
IMG_SIZE = 48
FACIAL_MODEL_PATH = "saved_models/facial_model.h5"
EEG_FILE = "Dataset/EEG Dataset/S1.mat"  # replace with your EEG file
EEG_THRESHOLD = 2.0  # P300 detection threshold

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
# LOAD FACE DETECTOR
# =========================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# =========================
# LOAD & PROCESS EEG
# =========================
if not os.path.exists(EEG_FILE):
    print("⚠️ EEG file not found, only facial stress will be used")
    eeg_available = False
    p300_probs = None
else:
    eeg_available = True
    data = sio.loadmat(EEG_FILE)
    eeg = data['test_data']  # (trials, samples, channels)
    trials, samples, channels = eeg.shape
    sfreq = 240
    ch_names = [f"EEG{i+1}" for i in range(channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    eeg = np.transpose(eeg, (0,2,1))  # (trials, channels, samples)
    epochs = mne.EpochsArray(eeg, info)

    # Use IIR filter to avoid FIR length warning
    epochs._data = mne.filter.filter_data(
        epochs._data, sfreq=sfreq, l_freq=0.1, h_freq=30, method='iir'
    )

    # Compute per-epoch P300 normalized probabilities
    p300_probs = []
    for ep_data in epochs._data:  # shape: (channels, samples)
        ep_avg = np.mean(ep_data, axis=0)  # avg across channels
        start = int(0.25*sfreq)
        end = int(0.5*sfreq)
        peak = np.max(ep_avg[start:end])
        prob = np.clip(peak / EEG_THRESHOLD, 0, 1)
        p300_probs.append(prob)
    p300_probs = np.array(p300_probs)
    print(f"✅ EEG processed: {len(p300_probs)} epochs")

# =========================
# START CAMERA
# =========================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

stress_scores = []
facial_scores = []
frame_index = 0
print("📷 Starting live facial stress detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face,(IMG_SIZE,IMG_SIZE))/255.0
        face = face.reshape(1,IMG_SIZE,IMG_SIZE,1)

        # Facial prediction
        pred = facial_model.predict(face, verbose=0)[0]
        emotion_index = np.argmax(pred)
        emotion = emotion_labels[emotion_index]

        facial_stress = sum(pred[i] for i,label in enumerate(emotion_labels) if label in stress_emotions)
        facial_stress = min(facial_stress,1.0)
        facial_scores.append(facial_stress)

        # EEG score per frame
        if eeg_available:
            eeg_score = p300_probs[frame_index % len(p300_probs)]
        else:
            eeg_score = 0.0

        # Fused stress
        fused = 0.7*facial_stress + 0.3*eeg_score
        stress_scores.append(fused)

        # Convert to percentage for display
        fused_percent = int(fused*100)
        facial_percent = int(facial_stress*100)
        eeg_percent = int(eeg_score*100)

        # ======================
        # DRAW OVERLAYS
        # ======================
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,f"Emotion: {emotion}",(x,y-50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv2.putText(frame,f"Facial Stress: {facial_percent}%",(x,y-30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
        cv2.putText(frame,f"EEG Stress: {eeg_percent}%",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
        cv2.putText(frame,f"Fused Stress: {fused_percent}%",(x,y+h+25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        # Stress bar
        bar_x, bar_y, bar_width, bar_height = 30, 50, 200, 20
        filled = int((fused_percent/100)*bar_width)
        cv2.rectangle(frame,(bar_x,bar_y),(bar_x+bar_width,bar_y+bar_height),(255,255,255),2)
        cv2.rectangle(frame,(bar_x,bar_y),(bar_x+filled,bar_y+bar_height),(0,0,255),-1)
        cv2.putText(frame,"Stress Level",(bar_x,bar_y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

    cv2.imshow("Cognitive Trace - EEG+Facial Fusion", frame)
    frame_index += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# =========================
# FINAL AVERAGE STRESS
# =========================
if stress_scores:
    avg_fused = np.mean(stress_scores)*100
    avg_facial = np.mean(facial_scores)*100
    avg_eeg = np.mean(p300_probs)*100 if eeg_available else 0.0

    print(f"\n📊 Average Fused Stress Score: {avg_fused:.2f}%")
    print(f"📈 Average Facial Stress: {avg_facial:.2f}%")
    if eeg_available:
        print(f"📈 Average EEG Stress: {avg_eeg:.2f}%")

    # Stress assessment
    if avg_fused > 60:
        print("⚠️ High Stress Detected")
    elif avg_fused > 40:
        print("⚠️ Moderate Stress")
    else:
        print("✅ Low Stress")
else:
    print("No faces detected")

# =========================
# PLOT EEG ERP
# =========================
if eeg_available:
    avg_signal = np.mean(epochs._data, axis=(0,1))  # avg across trials & channels
    times = np.arange(samples)/sfreq*1000
    plt.figure()
    plt.plot(times, avg_signal)
    plt.axvline(x=300, linestyle='--', color='r')
    plt.title("EEG ERP Waveform")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (µV)")
    plt.show()