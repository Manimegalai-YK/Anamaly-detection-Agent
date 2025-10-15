# at top of file
from notify import frame_to_jpeg_bytes, send_email_alert
import time, os, cv2

COOLDOWN = 60                      
last_alert_time = 0                
if err > THRESHOLD:
    now = time.time()
    if now - last_alert_time >= COOLDOWN:
        ts = time.strftime("%Y%m%d_%H%M%S")
        frame_marked = frame.copy()
        cv2.rectangle(frame_marked, (5,5), (frame_marked.shape[1]-5, frame_marked.shape[0]-5), (0,0,255), 4)
        cv2.putText(frame_marked, f"Anomaly Err:{err:.4f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        os.makedirs("alerts", exist_ok=True)
        local_path = os.path.join("alerts", f"anomaly_{ts}.jpg")
        cv2.imwrite(local_path, frame_marked)

        # convert to bytes and send email
        try:
            jpg = frame_to_jpeg_bytes(frame_marked)
            subject = f"[ALERT] Anomaly detected at {ts}"
            body = f"Anomaly detected at {ts}\nError: {err:.4f}\nThreshold: {THRESHOLD}\nSaved: {local_path}"
            ok = send_email_alert(subject, body, jpg, filename=f"anomaly_{ts}.jpg")
            if ok:
                last_alert_time = now
        except Exception as e:
            print("Notify error:", e)
    else:
        cv2.putText(frame, "ALERT (cooldown)", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)