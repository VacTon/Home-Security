
import cv2
import time
import logging
import numpy as np

# Try importing Picamera2 (The best way for Pi 5)
try:
    from picamera2 import Picamera2
    HAS_PICAM2 = True
except ImportError:
    HAS_PICAM2 = False
    logging.warning("Picamera2 not found. Falling back to OpenCV.")

class Camera:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.cap = None
        self.picam2 = None
        self.use_picam = False

    def start(self):
        """Starts the camera stream."""
        logging.info("Starting camera...")

        # 1. OPTION A: Picamera2 (Native Pi 5 support)
        if HAS_PICAM2:
            try:
                logging.info("Initializing Picamera2...")
                self.picam2 = Picamera2()
                
                # Configure for video capture
                config = self.picam2.create_video_configuration(
                    main={"size": (self.width, self.height), "format": "BGR888"}
                )
                self.picam2.configure(config)
                self.picam2.start()
                
                self.use_picam = True
                logging.info("Camera started successfully using Picamera2!")
                return
            except Exception as e:
                logging.error(f"Picamera2 failed to start: {e}")
                logging.warning("Falling back to OpenCV scan...")
                self.use_picam = False

        # 2. OPTION B: OpenCV Auto-Scan (Fallback)
        configs = [
            (0, cv2.CAP_V4L2, 'MJPG'),
            (0, cv2.CAP_V4L2, 'YUYV'),
            (0, cv2.CAP_ANY,  None),
            (1, cv2.CAP_V4L2, 'MJPG'),
            (1, cv2.CAP_V4L2, 'YUYV'),
        ]

        for idx, backend, fourcc in configs:
            backend_name = "V4L2" if backend == cv2.CAP_V4L2 else "ANY"
            fmt_name = fourcc if fourcc else "Default"
            logging.info(f"Testing OpenCV Index {idx} ({backend_name}, {fmt_name})...")
            
            try:
                cap = cv2.VideoCapture(idx, backend)
                if not cap.isOpened():
                    continue
                
                if fourcc:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
                
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                
                # Test Read
                for _ in range(5):
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        logging.info(f" -> Success! Using OpenCV Index {idx}")
                        self.cap = cap
                        return
                    time.sleep(0.1)
                cap.release()
            except:
                if 'cap' in locals(): cap.release()

        logging.error("Could not find any working camera.")
        raise RuntimeError("Camera failed")

    def get_frame(self):
        """Reads a frame from the camera."""
        if self.use_picam and self.picam2:
            # Capture directly to numpy array (very fast)
            return self.picam2.capture_array()
            
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None

    def release(self):
        if self.use_picam and self.picam2:
            self.picam2.stop()
            self.picam2 = None
        if self.cap:
            self.cap.release()
            self.cap = None

    def stop(self):
        self.release()
