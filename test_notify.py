# test_notify.py
import cv2
from notify import frame_to_jpeg_bytes, send_email_alert
import numpy as np
img = np.zeros((480,640,3), dtype='uint8') + 200
cv2.putText(img, "TEST ALERT", (50,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
jpg = frame_to_jpeg_bytes(img)
send_email_alert("Test Alert", "This is a test", jpg, "test.jpg")