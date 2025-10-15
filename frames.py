import numpy as np
import os
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Define autoencoder architecture (example)
input_img = Input(shape=(64, 64, 3))

x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d')(input_img)
x = MaxPooling2D((2, 2), padding='same', name='max_pooling2d')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2d_1')(x)
x = MaxPooling2D((2, 2), padding='same', name='max_pooling2d_1')(x)
encoded = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2d_2')(x)

x = UpSampling2D((2, 2), name='up_sampling2d')(encoded)
x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d_3')(x)
x = UpSampling2D((2, 2), name='up_sampling2d_1')(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='conv2d_4')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
# Here you should train your autoencoder with your dataset
# For example:
# autoencoder.fit(train_data, train_data, epochs=20, batch_size=32, validation_split=0.1)
# For now, let's save the untrained model (just for demonstration)
autoencoder.save('autoencoder.h5')
# Then load it back
autoencoder = load_model('autoencoder.h5')
# Extract encoder model
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('conv2d_2').output)
# Folder containing frame images
frame_folder = 'frames_folder'
features = []
for filename in sorted(os.listdir(frame_folder)):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(frame_folder, filename)
        img = load_img(img_path, target_size=(64,64))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        feat = encoder.predict(img_array)
        features.append(feat.flatten())

features = np.array(features)
print('Extracted features shape:', features.shape)
np.save('frames.npy', features)
