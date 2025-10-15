# 🏭 Production Line Anomaly Detection Agent

## 🎯 Goal

Automatically detect anomalies (errors, slowdowns, defects, unsafe actions) in a manufacturing or assembly line using video and sensor data — then alert the line manager in real time.

### ⚙ How It Works

🧩 Data Input

🎥 Video Feed — CCTV / IP Cameras capturing production line.

⚡ Sensor Data — temperature, vibration, motor speed, torque, etc.

### 🤖 Detection Methods

####Technique	Purpose

YOLO (You Only Look Once)	Detect and track objects (workers, parts, tools, PPE).
LSTM (Long Short-Term Memory)	Understand workflow patterns and motion sequences over time.
OpenCV	Analyze frame differences, object speed, and detect irregular movements.

### 🧠 Model Training

#### 1. Collect Normal Workflow Data
- Record videos of normal operations and collect sensor logs.

#### 2. Label Anomalies
Examples:

- Missing screw or part
- Worker skips assembly step
- No gloves or helmet
- Machine jam or product drop

#### 3. Train Models

YOLOv8 → for object and PPE detection.

LSTM → for time-based workflow anomaly prediction.

###⚡ Real-Time Monitoring

- 1. Capture incoming video frames in real time.
- 2. Use YOLO to detect objects & classify actions.
- 3. Feed detection sequences into LSTM.
- 4. Compare with learned “normal” patterns.
- 5. If deviation found → Trigger anomaly alert.
     
### 🚨 Notification System
   When anomaly is detected:
<img width="1832" height="791" alt="image" src="https://github.com/user-attachments/assets/7e896667-8df4-48ad-9acf-f7b7e79e172d" />
<img width="1827" height="793" alt="image" src="https://github.com/user-attachments/assets/944cd95f-ce64-46e0-bd87-cf7f6cd15dbd" />
<img width="1689" height="796" alt="image" src="https://github.com/user-attachments/assets/fbd2cc9f-4ae9-4199-8122-ccdfacefdcb5" />
<img width="1918" height="711" alt="image" src="https://github.com/user-attachments/assets/aa7123aa-ce21-4f84-b875-5933493f9c93" />

- 🔔 Dashboard Alert (Streamlit / Flask dashboard)
- 📩 Email / SMS Notification to line manager
- 🏭 Integration with factory monitoring system (e.g., SCADA / ERP)
  
### 🧭 Agent Workflow

flowchart TD
    A[Video/Sensor Input] --> B[YOLO Object Detection]
    B --> C[LSTM Action Recognition]
    C --> D{Normal or Anomaly?}
    D -- Normal --> E[Log as OK]
    D -- Anomaly --> F[Trigger Alert Agent]
    F --> G[Notify Line Manager]
    
###🧰 Tech Stack

Component	Tool / Framework

Object Detection	YOLOv8 / YOLOv5
Sequence Modeling	TensorFlow / PyTorch (LSTM, GRU)
Image Processing	OpenCV
Data Handling	NumPy, Pandas
Visualization	Streamlit / Dash
Alerts	Twilio / SMTP / Firebase
Integration	MQTT / REST API / WebSocket

### 📌 Example Use Cases

- 👷 Worker skips or reverses a process step.

- ⚙ Conveyor slows or stops unexpectedly.

- 📦 Product drops or misplaced.

- 🧤 PPE not detected (no gloves/helmet).

- 🛠 Machine produces defective part.

Linkedin = https://www.linkedin.com/in/manimegalai-yuvaraj-a40a092a0/
Email = manimegalaiyuvaraj@gmail.com
