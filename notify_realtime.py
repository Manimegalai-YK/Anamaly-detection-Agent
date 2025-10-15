# realtime.py
import os, time, smtplib
import numpy as np
import cv2
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from keras.models import load_model
from keras.losses import MeanSquaredError
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

EMAIL_SENDER   = "manimegalaiyuvaraj38@gmail.com"        
EMAIL_APP_PASS = "inmh fsky zoje ukjg"        
EMAIL_RECEIVER = "manimegalaiyuvaraj38@gmail.com"    
THRESHOLD = 0.01          
ALERT_COOLDOWN_SEC = 60  
_last_alert_ts = 0.0
MODEL_PATH_KERAS = "autoencoder_model.keras"
MODEL_PATH_H5    = "autoencoder_model.h5"
def load_autoencoder():
    if os.path.exists(MODEL_PATH_KERAS):
        print("[INFO] Loading:", MODEL_PATH_KERAS)
        return load_model(MODEL_PATH_KERAS)
    print("[INFO] Loading:", MODEL_PATH_H5)
    return load_model(MODEL_PATH_H5, custom_objects={"mse": MeanSquaredError()})
autoencoder = load_autoencoder()
scaler_mean = None
scaler_scale = None
if os.path.exists("scaler_mean.npy") and os.path.exists("scaler_scale.npy"):
    scaler_mean = np.load("scaler_mean.npy")
    scaler_scale = np.load("scaler_scale.npy")
    print("[INFO] Scaler loaded.")
else:
    print("[WARN] Scaler files not found. Proceeding without StandardScaler.")
def scale_features(vec1280: np.ndarray) -> np.ndarray:
    """vec1280: (1280,) float32"""
    if scaler_mean is not None and scaler_scale is not None:
        return (vec1280 - scaler_mean) / (scaler_scale + 1e-8)
    mu = vec1280.mean()
    sd = vec1280.std() + 1e-8
    return (vec1280 - mu) / sd
print("[INFO] Loading MobileNetV2 feature extractor...")
feature_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg") 
def frame_to_features(frame_bgr: np.ndarray) -> np.ndarray:
    """Return (1280,) float32 feature vector from BGR frame."""
    img = cv2.resize(frame_bgr, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = img.astype(np.float32)
    x = preprocess_input(x)          
    x = np.expand_dims(x, axis=0)    
    feats = feature_model.predict(x, verbose=0) 
    return feats[0].astype(np.float32)
def send_email_alert(mse_value: float):
    global _last_alert_ts
    now = time.time()
    if now - _last_alert_ts < ALERT_COOLDOWN_SEC:
        print("[EMAIL] Cooldown active. Skipping email.")
        return
    _last_alert_ts = now
    subject = "⚠ Anomaly Detected in Production Line"
    body = f"Anomaly detected.\nMSE: {mse_value:.6f}\nThreshold: {THRESHOLD}"
    msg = MIMEMultipart()
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as server:
            server.login(EMAIL_SENDER, EMAIL_APP_PASS)  
            server.send_message(msg)
        print("[EMAIL] Alert sent successfully.")
    except Exception as e:
        print("[EMAIL] Failed:", e)
def main():
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return
    print("[INFO] Real-time monitoring started. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        feats = frame_to_features(frame)          
        feats_scaled = scale_features(feats)       
        x = np.expand_dims(feats_scaled, axis=0)  
        recon = autoencoder.predict(x, verbose=0)
        mse = float(np.mean((x - recon) ** 2))
        txt = f"MSE: {mse:.5f}  Th: {THRESHOLD:.5f}"
        color = (0, 255, 0)
        if mse > THRESHOLD:
            color = (0, 0, 255)
            cv2.putText(frame, "ANOMALY!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            send_email_alert(mse)
        cv2.putText(frame, txt, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Production Line Monitoring", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
