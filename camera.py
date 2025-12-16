import cv2
import time
import logging

class Camera:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.cap = None

    def start(self):
        """Starts the camera stream using Libcamera (via OpenCV backend)."""
        logging.info("Starting camera...")
        # On Pi 5 with libcamera, index 0 usually works if legacy stack is off.
        # Sometimes GStreamer pipeline is needed, but let's try standard V4L2 first.
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self.cap.isOpened():
            logging.error("Could not open camera. Ensure libcamera is working.")
            raise RuntimeError("Could not open camera")

    def get_frame(self):
        """Reads a frame from the camera."""
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None

    def stop(self):
        if self.cap:
            self.cap.release()
