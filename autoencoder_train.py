import os
import numpy as np
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
features_dir = "features"
model_save_path_keras = "autoencoder_model.keras"  
model_save_path_h5 = "autoencoder_model.h5"        
print("[INFO] Loading features...")
X = []
for file in sorted(os.listdir(features_dir)):
    if file.endswith(".npy"):
        features = np.load(os.path.join(features_dir, file))
        X.append(features)
X = np.vstack(X)
print(f"[INFO] Features shape: {X.shape}")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
np.save("scaler_mean.npy", scaler.mean_)
np.save("scaler_scale.npy", scaler.scale_)
input_dim = X_scaled.shape[1]
encoding_dim = 64
input_layer = layers.Input(shape=(input_dim,))
encoded = layers.Dense(128, activation="relu")(input_layer)
encoded = layers.Dense(encoding_dim, activation="relu")(encoded)
decoded = layers.Dense(128, activation="relu")(encoded)
decoded = layers.Dense(input_dim, activation="linear")(decoded)
autoencoder = models.Model(input_layer, decoded)
autoencoder.compile(optimizer="adam", loss="mse")
print("[INFO] Training autoencoder...")
autoencoder.fit(
    X_scaled, X_scaled,
    epochs=50,
    batch_size=32,
    shuffle=True,
    validation_split=0.1
)
autoencoder.save(model_save_path_keras)
print(f"[INFO] Model saved to {model_save_path_keras}")
autoencoder.save(model_save_path_h5)
print(f"[INFO] Backup model saved to {model_save_path_h5}")
