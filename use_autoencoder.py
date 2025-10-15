import numpy as np
from tensorflow.keras.models import load_model
import os

# Model file உண்டு இல்லையா check பண்ணல்
if os.path.exists('autoencoder.keras'):
    autoencoder = load_model('autoencoder.keras')
    print("Model loaded successfully!")

    # Dummy test data (உன் data கொண்டு மாற்றிக் கொள்)
    test_frames = np.random.rand(10,64,64,3).astype('float32')

    reconstructed = autoencoder.predict(test_frames)
    print("Prediction done. Output shape:", reconstructed.shape)
else:
    print("Model file 'autoencoder.keras' கிடைக்கவில்லை. முதலில் training பண்ணி சேமிக்கவும்.")
