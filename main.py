import cv2
import yaml
import logging
import time
import os
from camera import Camera
from detector import Detector
from recognizer import Recognizer
from notifier import Notifier

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    
    # Initialize Modules
    try:
        camera = Camera(width=config["system"]["frame_width"], height=config["system"]["frame_height"])
        detector = Detector(config)
        recognizer = Recognizer(config)
        notifier = Notifier(config)
    except Exception as e:
        logging.error(f"Initialization failed: {e}")
        return

    # Load known faces
    recognizer.load_known_faces()

    # Start Camera
    try:
        camera.start()
    except RuntimeError:
        logging.error("Failed to start camera.")
        return

    logging.info("System started. Press 'q' to quit.")

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                logging.warning("Empty frame received.")
                time.sleep(0.1)
                continue

            # 1. Detect Faces (YOLO)
            detections = detector.detect(frame)

            # 2. Process Detections
            for det in detections:
                box = det["box"] # [x1, y1, x2, y2]
                x1, y1, x2, y2 = box
                kpts = det.get("keypoints")

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw Keypoints if available
                if kpts is not None:
                     for kp in kpts:
                         cv2.circle(frame, (int(kp[0]), int(kp[1])), 2, (0, 255, 255), -1)

                # 3. Recognize Face
                # Pass full frame + info for alignment
                name, conf = recognizer.identify(frame, kpts=kpts, box=box)
                
                # Draw Name
                label = f"{name} ({conf:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 4. Notify
                if name != "Unknown":
                    notifier.notify(name)
                elif name == "Unknown":
                    # Optional: Notify for strangers too
                    notifier.notify("Unknown")

            # Display
            cv2.imshow("Security Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logging.info("Stopping...")
    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
