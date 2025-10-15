import cv2
import time
import os
from datetime import datetime
os.makedirs("dataset/normal", exist_ok=True)
os.makedirs("dataset/abnormal", exist_ok=True)
camera_index = 0
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print(f"ERROR: Cannot open camera index {camera_index}.")
    exit()
print(f"Camera opened at index {camera_index}. Keys: 'n'=normal, 'a'=abnormal, 'q'=quit")
recording = False
start_time = None
out = None
max_duration = 10  # seconds
while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to read frame.")
        break
    key = cv2.waitKey(1) & 0xFF
    if key == ord('n') and not recording:
        filename = f"dataset/normal/normal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 20, (frame.shape[1], frame.shape[0]))
        recording = "normal"
        start_time = time.time()
        print(f"Started recording NORMAL -> {filename}")
    elif key == ord('a') and not recording:
        filename = f"dataset/abnormal/abnormal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 20, (frame.shape[1], frame.shape[0]))
        recording = "abnormal"
        start_time = time.time()
        print(f" Started recording ABNORMAL -> {filename}")
    if recording:
        out.write(frame)
        elapsed = time.time() - start_time
        if elapsed >= max_duration:
            out.release()
            print(f" Auto-stopped {recording} recording ({max_duration} sec)")
            recording = False
    cv2.imshow("Record Camera Clips", frame)
    if key == ord('q'):
        print(" Exiting...")
        break
if recording and out:
    out.release()
cap.release()
cv2.destroyAllWindows()