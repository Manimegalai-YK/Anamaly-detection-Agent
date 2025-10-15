import cv2
print("Checking available camera indexes...\n")
for i in range(5):  # Check first 5 indexes
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        cap.release()
    else:
        print(f"No camera at index {i}")