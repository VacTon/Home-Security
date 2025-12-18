import cv2
import time
import logging

class Camera:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.cap = None

    def start(self):
        """Starts the camera stream with auto-discovery."""
        logging.info("Starting camera...")
        
        # Configurations to try (Index, Backend, FourCC)
        # Pi 5 often has video0 as meta and video1 as data? Or index 0 works with MJPG.
        configs = [
            (0, cv2.CAP_V4L2, 'MJPG'),
            (0, cv2.CAP_V4L2, 'YUYV'),
            (0, cv2.CAP_ANY,  None),
            (1, cv2.CAP_V4L2, 'MJPG'),
            (1, cv2.CAP_V4L2, 'YUYV'),
            (1, cv2.CAP_ANY,  None),
        ]

        for idx, backend, fourcc in configs:
            backend_name = "V4L2" if backend == cv2.CAP_V4L2 else "ANY"
            fmt_name = fourcc if fourcc else "Default"
            logging.info(f"Testing Camera Index {idx} with {backend_name} + {fmt_name}...")
            
            try:
                cap = cv2.VideoCapture(idx, backend)
                if not cap.isOpened():
                    continue
                
                if fourcc:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
                
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                
                # Warmup / Test Read
                success = False
                for _ in range(10): # Try reading a few frames
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        success = True
                        break
                    time.sleep(0.1)
                
                if success:
                    logging.info(f" -> Success! Using Camera Index {idx} ({backend_name}, {fmt_name})")
                    self.cap = cap
                    return
                else:
                    cap.release()
                    
            except Exception as e:
                logging.warning(f"Error testing config: {e}")
                if 'cap' in locals(): cap.release()

        # If we get here, nothing worked.
        logging.error("Could not find a working camera configuration.")
        logging.error("Troubleshooting: Run 'rpicam-hello -t 5' to verify hardware. Ensure no other app is using the camera.")
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
