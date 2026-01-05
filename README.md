# 🏠 AI-Powered Home Security System (Raspberry Pi 5)

A real-time face recognition security system running on Raspberry Pi 5 with Camera Module 3. Features face detection, recognition, mesh visualization, and instant Telegram alerts for unauthorized access.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%205-red.svg)
![FPS](https://img.shields.io/badge/FPS-18--20-green.svg)

## 🎯 Features

- **Real-time Face Detection**: MediaPipe Face Detection optimized for CPU (8-10ms)
- **Face Recognition**: Pre-trained ArcFace model with 512-dimensional embeddings
- **Face Mesh Visualization**: Cyan wireframe overlay with nose crosshair tracking
- **Identity Deduplication**: Prevents multiple faces from being assigned the same identity
- **Instant Telegram Alerts**: Photos sent when strangers are detected (10-second cooldown)
- **Threaded Architecture**: Background recognition for smooth 18-20 FPS performance
- **Data Collection Tools**: Automated burst capture (150 photos in 10 seconds)
- **Privacy-Focused**: All processing happens locally on your Raspberry Pi

## 🛠️ Hardware Requirements

- Raspberry Pi 5 (4GB+ RAM recommended)
- Raspberry Pi Camera Module 3
- MicroSD Card (32GB+ recommended)
- Power Supply (27W USB-C recommended)

## 📦 Software Requirements

- Raspberry Pi OS (64-bit)
- Python 3.9+
- Picamera2 library
- See `requirements.txt` for full dependencies

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/VacTon/Home-Security.git
cd Home-Security
git checkout v2-custom-training
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Pre-trained Model

Download the ArcFace model and place it in the `models/` directory:

- **ArcFace**: `w600k_r50.onnx` - [Download Link](https://github.com/onnx/models/tree/main/vision/body_analysis/arcface)

### 4. Set Up Telegram Bot

1. Open Telegram and search for **@BotFather**
2. Send `/newbot` and follow the prompts
3. Save your **bot token** (looks like `123456789:ABC...`)
4. Get your **chat ID** from **@userinfobot**

### 5. Configure the System

Edit `config.yaml`:

```yaml
notification:
  telegram_bot_token: "YOUR_BOT_TOKEN_HERE"
  telegram_chat_id: "YOUR_CHAT_ID_HERE"

system:
  confidence_threshold: 0.5
  recognition_tolerance: 0.40
  frame_width: 640
  frame_height: 480
```

### 6. Add Known Faces

**Option A: Manual Photos**

Create directory structure:

```
faces/
├── Person1/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
└── Person2/
    ├── photo1.jpg
    └── photo2.jpg
```

**Option B: Automated Capture (Recommended)**

```bash
python tools/add_user.py
```

Follow the prompts to capture 150 photos in 10 seconds.

### 7. Generate Face Database

```bash
python tools/process_database.py
```

This creates `models/known_faces.pkl` with face embeddings.

### 8. Run the System

```bash
python main.py
```

Press `q` to quit.

## 📁 Project Structure

```
Home-Security/
├── main.py                    # Main application (threaded architecture)
├── camera.py                  # Camera interface (Picamera2)
├── detector.py                # Face detection (MediaPipe)
├── recognizer.py              # Face recognition (ArcFace)
├── visualizer.py              # Face Mesh visualization
├── notifier.py                # Telegram notifications
├── config.yaml                # Configuration file
├── requirements.txt           # Python dependencies
├── tools/
│   ├── add_user.py           # Automated photo capture (150 images)
│   └── process_database.py   # Face database generator
├── models/
│   ├── w600k_r50.onnx        # ArcFace model (download separately)
│   └── known_faces.pkl       # Generated face embeddings
├── faces/                     # Known faces directory
│   └── [PersonName]/         # Add photos here
└── strangers/                 # Auto-saved stranger photos
```

## ⚙️ How It Works

1. **Camera Capture**: Picamera2 captures video at 640x480 resolution
2. **Face Detection**: MediaPipe detects faces (8-10ms per frame)
3. **Face Mesh**: MediaPipe draws wireframe overlay on detected faces
4. **Face Recognition**: ArcFace generates embeddings in background thread (150ms)
5. **Identity Deduplication**: Resolves conflicts when multiple faces match same identity
6. **Identification**: Cosine similarity matching against known faces database
7. **Alert System**: Telegram message with photo sent when stranger detected

## 🎨 Performance Optimization

Our system achieves **18-20 FPS** through three key optimizations:

1. **Detector Replacement**: Switched from YOLO (550ms) to MediaPipe (10ms) - 50x speedup
2. **ROI Processing**: Face Mesh processes only detected face region, not full frame
3. **Threading**: Recognition runs in background thread (AsyncRecognizer class)

### Performance Metrics (Raspberry Pi 5)

- **Single Person**: 20-25 FPS
- **Three People**: 11-15 FPS
- **Detection Latency**: 8-10ms (MediaPipe)
- **Recognition Latency**: 150ms (background thread, non-blocking)
- **Mesh Rendering**: 30ms per face
- **Memory Usage**: ~600MB
- **CPU Usage**: 50-70% (multi-core)

## 🔧 Configuration Options

### System Settings

- `confidence_threshold`: Minimum confidence for face detection (0.0-1.0)
- `recognition_tolerance`: Similarity threshold for face matching (0.40 recommended)
- `frame_width/height`: Camera resolution (640x480 recommended)

### Telegram Settings

- `telegram_bot_token`: Your Telegram bot token from @BotFather
- `telegram_chat_id`: Your Telegram chat ID

## 📸 Stranger Detection

When an unknown person is detected:
1. A snapshot is saved to `strangers/stranger_YYYYMMDD_HHMMSS.jpg`
2. A Telegram message is sent with the photo
3. The system waits 10 seconds before sending another alert (cooldown)

## 🔧 Tools

### add_user.py - Automated Data Collection

Captures 150 photos in 10 seconds for training:

```bash
python tools/add_user.py
```

- Enter user name when prompted
- Position face in camera view
- Press ENTER to start capture
- Rotate head slowly during capture
- Photos saved to `faces/[Name]/`

### process_database.py - Database Generation

Processes all photos in `faces/` directory and generates embeddings:

```bash
python tools/process_database.py
```

- Detects faces in each photo
- Generates 512-dimensional embeddings
- Saves to `models/known_faces.pkl`
- Run this after adding new users

## 🐛 Troubleshooting

### Camera Issues
```bash
# Check if camera is detected
libcamera-hello
```

### Low FPS
- System is optimized for 18-20 FPS
- FPS drops with multiple faces (expected behavior)
- Ensure good lighting for faster processing

### Recognition Not Working
- Re-run `python tools/process_database.py`
- Check `recognition_tolerance` in config.yaml (0.40 recommended)
- Add more photos using `tools/add_user.py`
- Ensure good lighting conditions

### Duplicate Names Appearing
- System includes deduplication logic (highest confidence wins)
- If still occurring, increase `recognition_tolerance` to 0.50

### Telegram Not Working
- Verify bot token and chat ID in config.yaml
- Check internet connection
- Test bot: `https://api.telegram.org/bot<TOKEN>/getMe`
- Send `/start` to your bot first

## 🌐 WiFi Configuration (For Demos)

To connect to multiple WiFi networks (e.g., home + mobile hotspot):

```bash
# Add new network
sudo nmcli device wifi connect "NetworkName" password "password"

# Set priority (higher = preferred)
sudo nmcli connection modify "NetworkName" connection.autoconnect-priority 10

# Verify saved networks
nmcli connection show
```

## 🔒 Privacy & Security

- All processing happens **locally** on your Raspberry Pi
- No cloud services required (except Telegram for notifications)
- Face embeddings stored locally in `models/known_faces.pkl`
- Stranger photos saved locally in `strangers/` directory
- No video uploaded to external servers

## 🙏 Acknowledgments

- [MediaPipe](https://github.com/google/mediapipe) for face detection and mesh
- [ArcFace](https://github.com/deepinsight/insightface) for face recognition
- [ONNX Runtime](https://onnxruntime.ai/) for efficient inference
- [Picamera2](https://github.com/raspberrypi/picamera2) for camera interface

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Team VacTon - CSE2022 Capstone Project**
