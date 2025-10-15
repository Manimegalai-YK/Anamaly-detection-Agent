import cv2
import os
import numpy as np
from ultralytics import YOLO

video_dir = "videos"       
features_dir = "features"  
os.makedirs(features_dir, exist_ok=True)

# Load YOLO model
model = YOLO("yolov8n.pt")

print("[INFO] Extracting features...")
for video_file in os.listdir(video_dir):
    if not video_file.lower().endswith((".mp4", ".avi", ".mov")):
        continue

    video_path = os.path.join(video_dir, video_file)
    cap = cv2.VideoCapture(video_path)

    video_features = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results[0].boxes.data.cpu().numpy() if results else []

        # Feature vector: [num_objects, avg_confidence]
        if len(detections) > 0:
            num_objects = len(detections)
            avg_conf = np.mean(detections[:, 4])
        else:
            num_objects = 0
            avg_conf = 0

        feature_vector = [num_objects, avg_conf]
        video_features.append(feature_vector)

    cap.release()

    video_features = np.array(video_features)
    feature_file = os.path.join(features_dir, f"{os.path.splitext(video_file)[0]}.npy")
    np.save(feature_file, video_features)

    print(f"[INFO] Saved features: {feature_file}")

print("[DONE] Feature extraction complete.")