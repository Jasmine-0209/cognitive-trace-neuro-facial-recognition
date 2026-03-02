import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# =====================================================
# PATH SETUP (SAFE FOR WINDOWS & ANY LOCATION)
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset", "CK+48")
MODEL_DIR = os.path.join(BASE_DIR, "..", "saved_models")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "facial_model.h5")

IMG_SIZE = 48

# Create saved_models folder if not exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# =====================================================
# CHECK DATASET
# =====================================================

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")

emotion_labels = sorted([
    folder for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
])

print("Detected Emotion Classes:", emotion_labels)

# =====================================================
# LOAD DATA
# =====================================================

data = []
labels = []

for label_index, emotion in enumerate(emotion_labels):
    emotion_path = os.path.join(DATASET_PATH, emotion)

    for img_name in os.listdir(emotion_path):
        img_path = os.path.join(emotion_path, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        data.append(img)
        labels.append(label_index)

data = np.array(data)
labels = np.array(labels)

data = data.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
labels = to_categorical(labels)

print("Total Images Loaded:", len(data))

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# =====================================================
# MODEL ARCHITECTURE
# =====================================================

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(len(emotion_labels), activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =====================================================
# TRAIN MODEL
# =====================================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    callbacks=[early_stop]
)

# =====================================================
# SAVE MODEL
# =====================================================

model.save(MODEL_SAVE_PATH)

print("\n✅ Training Complete!")
print(f"Model saved at: {MODEL_SAVE_PATH}")
