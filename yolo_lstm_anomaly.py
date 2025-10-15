import os
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
VIDEO_PATH = "production_line.mp4"  
SEQUENCE_LENGTH = 10 
FEATURE_SIZE = 8400 
print("[INFO] Loading YOLOv8 model...")
yolo_model = YOLO("yolov8n.pt") 
def extract_features(video_path):
    cap = cv2.VideoCapture(video_path)
    features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame, verbose=False)

        if hasattr(results[0], "boxes") and results[0].boxes is not None:
            feat_vector = results[0].boxes.xywh.cpu().numpy().flatten()
        else:
            feat_vector = np.zeros(FEATURE_SIZE)

        if feat_vector.shape[0] < FEATURE_SIZE:
            feat_vector = np.pad(feat_vector, (0, FEATURE_SIZE - feat_vector.shape[0]))
        else:
            feat_vector = feat_vector[:FEATURE_SIZE]

        features.append(feat_vector)

    cap.release()
    return np.array(features)

print("[INFO] Extracting YOLO features from video...")
raw_features = extract_features(VIDEO_PATH)
print(f"[INFO] Extracted features shape: {raw_features.shape}")

def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences)

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(raw_features)

X = create_sequences(scaled_features, SEQUENCE_LENGTH)
y = np.zeros((X.shape[0],)) 
print(f"[INFO] LSTM input shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

print("[INFO] Training LSTM model...")
model.fit(X_train, y_train, epochs=5, batch_size=8, validation_data=(X_test, y_test))

model.save("yolo_lstm_anomaly.h5")
print("[INFO] Model saved to yolo_lstm_anomaly.h5")
