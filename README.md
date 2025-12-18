# 🏠 Raspberry Pi 5 Face Recognition Security System

A real-time face recognition security system built for Raspberry Pi 5 with Camera Module 3, featuring instant Telegram notifications with photos when strangers are detected.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%205-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Features

- **Real-time Face Detection**: YOLOv8-Face for fast and accurate face detection
- **Face Recognition**: ArcFace model with ONNX Runtime for reliable identification
- **Instant Alerts**: Telegram bot notifications with photos when strangers are detected
- **Smart Tracking**: Frame-skipping optimization for smooth 25-30 FPS performance
- **Easy Setup**: Simple configuration via YAML file
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
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Pre-trained Models

Download the following models and place them in the `models/` directory:

- **YOLOv8-Face**: `yolov8n-face.onnx` - [Download Link](https://github.com/derronqi/yolov8-face)
- **ArcFace**: `w600k_r50.onnx` - [Download Link](https://github.com/onnx/models/tree/main/vision/body_analysis/arcface)

### 4. Set Up Telegram Bot

1. Open Telegram and search for **@BotFather**
2. Send `/newbot` and follow the prompts
3. Save your **bot token** (looks like `123456789:ABC...`)
4. Get your **chat ID** from **@userinfobot**

### 5. Configure the System

Edit `config.yaml`:

```yaml
telegram:
  enabled: true
  bot_token: "YOUR_BOT_TOKEN_HERE"
  chat_id: "YOUR_CHAT_ID_HERE"
  cooldown_seconds: 60

system:
  confidence_threshold: 0.5
  recognition_tolerance: 0.4
  frame_width: 640
  frame_height: 480
```

### 6. Add Known Faces

Create a directory structure like this:

```
faces/
├── Home_Owners/
│   ├── Person1/
│   │   ├── photo1.jpg
│   │   ├── photo2.jpg
│   │   └── photo3.jpg
│   └── Person2/
│       ├── photo1.jpg
│       └── photo2.jpg
```

**Tips for best results:**
- Use 3-5 photos per person
- Include different angles and lighting
- Ensure faces are clearly visible
- Photos should be well-lit

### 7. Generate Face Database

```bash
python tools/process_database.py
```

This will create `faces/encodings.pkl` with all known face embeddings.

### 8. Run the System

```bash
python main.py
```

Press `q` to quit.

## 📁 Project Structure

```
Home-Security/
├── main.py                 # Main application entry point
├── camera.py              # Camera interface (Picamera2)
├── detector.py            # Face detection (YOLOv8)
├── recognizer.py          # Face recognition (ArcFace)
├── notifier.py            # Telegram notifications
├── config.yaml            # Configuration file
├── requirements.txt       # Python dependencies
├── tools/
│   └── process_database.py  # Face database generator
├── models/                # Pre-trained models (download separately)
├── faces/                 # Known faces directory
│   ├── Home_Owners/       # Add your photos here
│   └── encodings.pkl      # Generated face embeddings
└── strangers/             # Auto-saved stranger photos
```

## ⚙️ How It Works

1. **Camera Capture**: Picamera2 captures video at 640x480 resolution
2. **Face Detection**: YOLOv8-Face detects faces in each frame (~30 FPS)
3. **Face Recognition**: ArcFace generates embeddings every 30 frames (1 sec intervals)
4. **Identification**: Cosine similarity matching against known faces database
5. **Alert System**: When a stranger is detected, a Telegram message with photo is sent

## 🎨 Performance Optimization

- **Frame Skipping**: Recognition runs every 30 frames to maintain high FPS
- **Face Tracking**: Simple center-based tracking to display names between recognition frames
- **CPU Inference**: ONNX Runtime optimized for ARM64 architecture
- **Efficient Preprocessing**: Face alignment using detected keypoints

## 🔧 Configuration Options

### System Settings

- `confidence_threshold`: Minimum confidence for face detection (0.0-1.0)
- `recognition_tolerance`: Similarity threshold for face matching (0.3-0.5)
- `frame_width/height`: Camera resolution

### Telegram Settings

- `enabled`: Enable/disable Telegram notifications
- `bot_token`: Your Telegram bot token from @BotFather
- `chat_id`: Your Telegram chat ID
- `cooldown_seconds`: Minimum time between notifications for the same person

## 📸 Stranger Detection

When an unknown person is detected:
1. A snapshot is saved to `strangers/stranger_YYYYMMDD_HHMMSS.jpg`
2. A Telegram message is sent with the photo
3. The system waits for the cooldown period before sending another alert

## 🐛 Troubleshooting

### Camera Issues
```bash
# Check if camera is detected
libcamera-hello
```

### Low FPS
- Reduce `frame_width` and `frame_height` in config.yaml
- Increase recognition interval in main.py (currently 30 frames)

### Recognition Not Working
- Re-run `python tools/process_database.py`
- Adjust `recognition_tolerance` in config.yaml (lower = more lenient)
- Ensure good lighting conditions

### Telegram Not Working
- Verify bot token and chat ID
- Check internet connection
- Test bot manually: `https://api.telegram.org/bot<TOKEN>/getMe`

## 📊 Performance Metrics

On Raspberry Pi 5 (4GB):
- **Detection FPS**: 25-30 FPS
- **Recognition Latency**: ~100ms per face
- **Memory Usage**: ~500MB
- **CPU Usage**: 40-60% (single core)

## 🔒 Privacy & Security

- All processing happens **locally** on your Raspberry Pi
- No cloud services required (except Telegram for notifications)
- Face embeddings are stored locally in `faces/encodings.pkl`
- Stranger photos are saved locally in `strangers/` directory

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [YOLOv8-Face](https://github.com/derronqi/yolov8-face) for face detection
- [ArcFace](https://github.com/deepinsight/insightface) for face recognition
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO implementation
- [ONNX Runtime](https://onnxruntime.ai/) for efficient inference

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Made with ❤️ for Raspberry Pi 5**
