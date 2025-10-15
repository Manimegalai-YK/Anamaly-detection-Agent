import os
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
frame_dir = 'frames'
categories = ['normal', 'abnormal']
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)
X = []
y = []
for label, category in enumerate(categories):
    category_path = os.path.join(frame_dir, category)
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        try:
            image = load_img(img_path, target_size=(224, 224))
            image = img_to_array(image)
            image = preprocess_input(image)
            image = np.expand_dims(image, axis=0)

            feature = model.predict(image, verbose=0)
            X.append(feature[0])
            y.append(label)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
X = np.array(X)
y = np.array(y)
np.save('features.npy', X)
np.save('labels.npy', y)
print("Feature extraction and dataset preparation complete!")
