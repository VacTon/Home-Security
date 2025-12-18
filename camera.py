
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
        """Starts the camera stream using Picamera2."""
        logging.info("Starting camera...")

        if not HAS_PICAM2:
            raise RuntimeError("Picamera2 is required for Raspberry Pi 5. Please install: pip install picamera2")

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
            
        except Exception as e:
            logging.error(f"Picamera2 failed to start: {e}")
            raise RuntimeError(f"Camera initialization failed: {e}")


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
