import numpy as np
import os
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

# Set the features folder
features_path = "features"

# Load all normal feature files
normal_features_list = []
for file in os.listdir(features_path):
    if file.startswith("normal") and file.endswith(".npy"):
        path = os.path.join(features_path, file)
        print(f"Loading normal feature file: {file}")
        features = np.load(path)
        normal_features_list.append(features)

# Combine all normal features into one array
normal_features = np.vstack(normal_features_list)

# Load all abnormal feature files
abnormal_features_list = []
for file in os.listdir(features_path):
    if file.startswith("abnormal") and file.endswith(".npy"):
        path = os.path.join(features_path, file)
        print(f"Loading abnormal feature file: {file}")
        features = np.load(path)
        abnormal_features_list.append(features)

# Combine all abnormal features
abnormal_features = np.vstack(abnormal_features_list)

# Train Isolation Forest on only normal features
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(normal_features)

# Predict on both normal and abnormal
normal_preds = model.predict(normal_features)      # +1 = normal
abnormal_preds = model.predict(abnormal_features)  # -1 = anomaly

# Prepare labels and predictions
y_true = np.concatenate([
    np.ones(len(normal_preds)),        # 1 = normal
    np.zeros(len(abnormal_preds))      # 0 = abnormal
])

y_pred = np.concatenate([
    (normal_preds == 1).astype(int),
    (abnormal_preds == 1).astype(int)
])

# Evaluation
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Abnormal", "Normal"]))
