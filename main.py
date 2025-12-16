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

            # 1. Detect Faces (using YOLO)
            detections = detector.detect(frame)

            # 2. Process Detections
            for det in detections:
                box = det["box"] # [x1, y1, x2, y2]
                x1, y1, x2, y2 = box

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Extract Face ROI
                # Ensure coordinates are within frame
                h, w, _ = frame.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                face_img = frame[y1:y2, x1:x2]
                
                if face_img.size == 0:
                    continue
                
                # Convert BGR to RGB for face_recognition
                face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

                # 3. Recognize Face
                name = recognizer.identify(face_img_rgb)
                
                # Draw Name
                cv2.putText(frame, f"{name} ({det['conf']:.2f})", (x1, y1 - 10), 
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
