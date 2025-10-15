# notify.py
import os, smtplib
from email.message import EmailMessage
import cv2

EMAIL_USER = os.environ.get("EMAIL_USER")    # set with setx or export
EMAIL_PASS = os.environ.get("EMAIL_PASS")
ALERT_TO   = os.environ.get("ALERT_TO")

def frame_to_jpeg_bytes(frame_bgr, quality=80):
    ok, buf = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode frame to JPEG")
    return buf.tobytes()

def send_email_alert(subject, body, jpeg_bytes=None, filename="anomaly.jpg"):
    if not (EMAIL_USER and EMAIL_PASS and ALERT_TO):
        print("Email env vars missing. Set EMAIL_USER, EMAIL_PASS, ALERT_TO.")
        return False

    msg = EmailMessage()
    msg["From"] = EMAIL_USER
    msg["To"] = ALERT_TO
    msg["Subject"] = subject
    msg.set_content(body)
    if jpeg_bytes:
        msg.add_attachment(jpeg_bytes, maintype="image", subtype="jpeg", filename=filename)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(EMAIL_USER, EMAIL_PASS)
            smtp.send_message(msg)
        print(f"Email sent to {ALERT_TO}")
        return True
    except Exception as e:
        print("Email failed:", e)
        return False
