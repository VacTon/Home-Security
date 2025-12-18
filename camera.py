import cv2
import time
import logging

class Camera:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.cap = None

    def start(self):
        """Starts the camera stream."""
        logging.info("Starting camera...")
        
        # 1. Try V4L2 Backend (Best for Pi 5 + Pip OpenCV)
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            if self._check_opened():
                logging.info("Camera opened with V4L2 backend.")
                return
        except Exception:
            pass

        # 2. Try Default Backend (Fallback)
        logging.warning("V4L2 backend failed. Trying default...")
        self.cap = cv2.VideoCapture(0)
        if self._check_opened():
             logging.info("Camera opened with default backend.")
             return

        # 3. Fail
        logging.error("Could not open camera! Check connection or try 'libcamera-hello' in terminal.")
        raise RuntimeError("Could not open camera")

    def _check_opened(self):
        if self.cap and self.cap.isOpened():
            # Apply Settings
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            return True
        return False

    def get_frame(self):
        """Reads a frame from the camera."""
        if self.cap:
             # Add a small delay if reading too fast? No.
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None

    def release(self):
        """Releases the camera resource."""
        if self.cap:
            self.cap.release()
            self.cap = None

    def stop(self):
        self.release()
