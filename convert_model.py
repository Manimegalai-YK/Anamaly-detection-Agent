import tensorflow as tf

# Load your existing .keras model
model = tf.keras.models.load_model("autoencoder.keras")
model.save("anomaly_detector_autoencoder.h5")
print("Model saved as anomaly_detector_autoencoder.h5")
