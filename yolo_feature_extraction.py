import os
import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

abnormal_path = "scripts/dataset/abnormal"
normal_path = "scripts/dataset/normal"
results_path = "results"
os.makedirs(results_path, exist_ok=True) 

def load_frames_from_videos(folder_path, max_frames=200, return_file_refs=False):
    frames = []
    file_refs = [] 
    for file in os.listdir(folder_path):
        if file.endswith(".mp4"):
            video_path = os.path.join(folder_path, file)
            cap = cv2.VideoCapture(video_path)
            count = 0
            while cap.isOpened() and count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                resized_frame = cv2.resize(frame, (64, 64))  
                resized_frame = resized_frame.astype("float32") / 255.0  # normalize
                frames.append(resized_frame)
                file_refs.append((file, count, frame)) 
                count += 1
            cap.release()
    if return_file_refs:
        return np.array(frames), file_refs
    return np.array(frames)
normal_frames = load_frames_from_videos(normal_path)
abnormal_frames, abnormal_refs = load_frames_from_videos(abnormal_path, return_file_refs=True)

print(f"Normal frames: {normal_frames.shape}")
print(f"Abnormal frames: {abnormal_frames.shape}")

input_img = Input(shape=(64, 64, 3))

# Encoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer=Adam(), loss='mse')

autoencoder.summary()
history = autoencoder.fit(
    normal_frames, normal_frames,
    epochs=20,
    batch_size=32,
    shuffle=True,
    validation_split=0.1
)


reconstructed_abnormal = autoencoder.predict(abnormal_frames)
errors = np.mean(np.square(abnormal_frames - reconstructed_abnormal), axis=(1, 2, 3))
plt.hist(errors, bins=30)
plt.title("Reconstruction Error for Abnormal Frames")
plt.xlabel("Error")
plt.ylabel("Count")
plt.show()

# ==== Step 8: Example threshold for anomaly detection ====
threshold = np.mean(errors) + 2*np.std(errors)
print(f"Threshold: {threshold}")

# ==== Step 9: Flag anomalies and save abnormal frames with red borders ====
anomalies = errors > threshold
print(f"Detected {np.sum(anomalies)} anomalous frames out of {len(errors)}")

for idx, is_anomaly in enumerate(anomalies):
    if is_anomaly:
        video_name, frame_idx, original_frame = abnormal_refs[idx]
        
        # Draw a thick red border
        bordered_frame = original_frame.copy()
        color = (0, 0, 255)  # BGR red
        thickness = 5
        cv2.rectangle(bordered_frame, (0, 0), (bordered_frame.shape[1]-1, bordered_frame.shape[0]-1), color, thickness)
        
        # Save to results folder
        save_name = f"{video_name}_frame{frame_idx}_abnormal.jpg"
        cv2.imwrite(os.path.join(results_path, save_name), bordered_frame)

print(f"Saved abnormal frames with red borders in '{results_path}' folder.")
plt.show(block=False)
plt.pause(3)  
plt.close()
