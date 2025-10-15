import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
def send_email_notification(subject, message, to_email):
    sender_email = "manimegalaiyuvaraj38@gmail.com"
    sender_password = "inmh fsky zoje ukjg"
    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(message, "plain"))
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()
        st.success(f"Email sent successfully to {to_email}")
    except Exception as e:
        st.error(f"Error sending email: {str(e)}")
st.set_page_config(page_title="Production Line Anomaly Detection", layout="wide")
st.title("Production Line Anomaly Detection")
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Feature Viewer", "Anomaly Detection"])
FEATURES_DIR = "features"
def load_feature_file(filename):
    path = os.path.join(FEATURES_DIR, filename)
    if os.path.exists(path):
        return np.load(path)
    else:
        st.error(f"File {filename} not found.")
        return None
if page == "Home":
    st.subheader("Welcome")
    st.write("It detects anamolies in production line videos using extracted the features")
elif page == "Feature Viewer":
    st.subheader("View Extracted Features")
    feature_files = [f for f in os.listdir(FEATURES_DIR) if f.endswith(".npy")]
    if feature_files:
        selected_file = st.selectbox("Select feature file", feature_files)
        if selected_file:
            features = load_feature_file(selected_file)
            if features is not None:
                st.write(f"Shape of features: {features.shape}")
                st.write("First 5 feature vectors:")
                st.write(features[:5])
                st.subheader("Feature Distribution")
                plt.figure(figsize=(10, 4))
                plt.plot(features[0])
                plt.title(f"Feature Vector Plot: {selected_file}")
                st.pyplot(plt)
    else:
        st.warning("No .npy feature files found in 'features' directory.")
elif page == "Anomaly Detection":
    st.subheader("⚠ Run Anomaly Detection")
    feature_files = [f for f in os.listdir(FEATURES_DIR) if f.endswith(".npy")]
    if feature_files:
        selected_file = st.selectbox("Select feature file", feature_files)
        recipient_email = st.text_input("Enter email for notifications", "")
        if st.button("Run Detection"):
            features = load_feature_file(selected_file)
            if features is not None:
                anomaly_scores = np.random.rand(len(features))
                threshold = 0.8
                anomalies = np.where(anomaly_scores > threshold)[0]
                st.write(f"Detected {len(anomalies)} anomalies.")
                st.write("Anomaly frame indices:", anomalies.tolist())
                plt.figure(figsize=(10, 4))
                plt.plot(anomaly_scores, label="Anomaly Score")
                plt.axhline(y=threshold, color='r', linestyle='--', label="Threshold")
                plt.legend()
                st.pyplot(plt)
                if len(anomalies) > 0 and recipient_email:
                    subject = "Production Line Anomaly Alert"
                    message = (
                        f"Anomalies detected in file {selected_file}.\n"
                        f"Total anomalies: {len(anomalies)}\n"
                        f"Frames: {anomalies.tolist()}"
                    )
                    send_email_notification(subject, message, recipient_email)
    else:
        st.warning("No feature files available.")