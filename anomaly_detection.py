import numpy as np
import cv2
import os
import tensorflow as tf
def load_frames(folder_path, limit=None):
    print(f"Loading frames from: {folder_path}")
    frames = []
    count = 0
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (64, 64))
            frames.append(img)
            count += 1
            if limit and count >= limit:
                break
    print(f"Loaded {len(frames)} frames from {folder_path}")
    return np.array(frames) / 255.0  # normalize
def calculate_errors(original_frames, reconstructed_frames):
    return np.mean((original_frames - reconstructed_frames) ** 2, axis=(1, 2, 3))
def detect_anomalies(new_original_frames, new_reconstructed_frames, threshold):
    new_errors = calculate_errors(new_original_frames, new_reconstructed_frames)
    anomaly_indices = np.where(new_errors > threshold)[0]
    return anomaly_indices, new_errors
def save_anomaly_frames(frames, anomaly_indices, save_folder="results"):
    os.makedirs(save_folder, exist_ok=True)
    for i in anomaly_indices:
        frame = (frames[i] * 255).astype(np.uint8).copy()
        frame = cv2.copyMakeBorder(frame, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[255, 0, 0])  # Red border
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_folder, f"anomaly_{i}.png"), frame)
    print(f"Saved {len(anomaly_indices)} anomaly frames to '{save_folder}'.")
def main():
    normal_folder = r"C:\Users\MANIMEGALAI\Desktop\Production line Anomaly detection\frames\normal1"
    abnormal_folder = r"C:\Users\MANIMEGALAI\Desktop\Production line Anomaly detection\frames\abnormal1"
    model_path = "autoencoder.keras"
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    normal_frames = load_frames(normal_folder, limit=500)
    abnormal_frames = load_frames(abnormal_folder, limit=500)
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
    # Calculate threshold from normal frames
    print("Predicting on normal frames...")
    normal_reconstructed = model.predict(normal_frames)
    print("Calculating errors...")
    normal_errors = calculate_errors(normal_frames, normal_reconstructed)
    threshold = np.percentile(normal_errors, 95)
    print(f"Calculated threshold: {threshold}")

    # Detect anomalies in abnormal frames
    print("Predicting on abnormal frames...")
    abnormal_reconstructed = model.predict(abnormal_frames)
    anomaly_idx, anomaly_errors = detect_anomalies(abnormal_frames, abnormal_reconstructed, threshold)
    print(f"Detected {len(anomaly_idx)} anomalous frames out of {len(abnormal_frames)}")

    # Save anomalous frames
    save_anomaly_frames(abnormal_frames, anomaly_idx)

if __name__ == "__main__":
    main()
