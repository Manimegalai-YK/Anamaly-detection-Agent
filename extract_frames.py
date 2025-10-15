import cv2
import os
input_dirs = {
    "normal": "dataset/normal",
    "abnormal": "dataset/abnormal"
}
output_base = "frames"
os.makedirs(output_base, exist_ok=True)
frames_per_second = 5 
frame_size = (128, 128)  # resize all frames

for label, input_path in input_dirs.items():
    output_path = os.path.join(output_base, label)
    os.makedirs(output_path, exist_ok=True)
    
    for video_file in os.listdir(input_path):
        if video_file.lower().endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(input_path, video_file)
            cap = cv2.VideoCapture(video_path)
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            interval = int(fps / frames_per_second)
            count = 0
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if count % interval == 0:
                    resized_frame = cv2.resize(frame, frame_size)
                    frame_filename = f"{os.path.splitext(video_file)[0]}_{frame_count}.jpg"
                    cv2.imwrite(os.path.join(output_path, frame_filename), resized_frame)
                    frame_count += 1
                
                count += 1
            
            cap.release()
            print(f"Extracted {frame_count} frames from {video_file}")

print("Frame extraction completed.")
