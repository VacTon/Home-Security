# Pi 5 Face Recognition Security System

This project implements a face recognition security system for Raspberry Pi 5.
It uses **ONNX Runtime** for high performance inference without requiring complex Hailo compilation (though Hailo can accelerate ONNX if configured, we use CPU/ONNXRuntime for simplicity conformant with Pi 5 specs).

## Features
- **Face Detection**: Uses YOLOv8-Face (ONNX) for fast detection with landmarks.
- **Face Recognition**: Uses ArcFace (ResNet50) ONNX for state-of-the-art embedding generation.
- **Notifications**: Email alerts when known or unknown faces are seen.

## Setup Instructions

### 1. Hardware
- Raspberry Pi 5
- Camera Module 3

### 2. Software Installation (on Pi 5)
```bash
sudo apt-get update
sudo apt-get install python3-opencv
pip install -r requirements.txt
```

### 3. Model Preparation
You need to obtain the ONNX models. You can generate them using the provided script (useful for running on Kaggle or a PC):

1. Run `training/kaggle_export.py` (e.g. on Kaggle Notebook).
2. It will download/export:
   - `yolov8n-face.onnx` (Detector)
   - `w600k_r50.onnx` (Recognizer - ArcFace)
3. Transfer these files to the `models/` directory on your Pi.
   - `pi_hailo_security/models/yolov8n-face.onnx`
   - `pi_hailo_security/models/w600k_r50.onnx`

### 4. Enroll Faces
Create a directory structure in `faces/`:
```
faces/
  Dad/
    photo1.jpg (Ideally a cropped face)
  Mom/
    photo2.jpg
```
The system will auto-scan these on startup.

### 5. Run
```bash
python main.py
```

## GitHub Integration
To link this to your GitHub:
1. Create a repository on GitHub.
2. Run locally:
```bash
git remote add origin https://github.com/YOUR_USERNAME/pi-face-security.git
git branch -M main
git push -u origin main
```
