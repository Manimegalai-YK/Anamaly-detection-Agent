import os
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tqdm import tqdm
frames_root = "frames"
features_root = "features"
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)
os.makedirs(features_root, exist_ok=True)
for video_folder in os.listdir(frames_root):
    video_path = os.path.join(frames_root, video_folder)
    if not os.path.isdir(video_path):
        continue
    print(f"Extracting features from: {video_folder}")
    features = []
    for img_file in tqdm(sorted(os.listdir(video_path))):
        if not img_file.endswith(".jpg"):
            continue
        img_path = os.path.join(video_path, img_file)
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = model.predict(x, verbose=0)
        features.append(feature.squeeze())
    features = np.array(features)
    save_path = os.path.join(features_root, f"{video_folder}.npy")
    np.save(save_path, features)
    print(f"Saved features to {save_path}")
