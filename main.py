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

    # Stranger Snapshot coodown
    last_stranger_shot = 0
    stranger_cooldown = 15.0 # seconds
    stranger_dir = "strangers"
    if not os.path.exists(stranger_dir):
        os.makedirs(stranger_dir, exist_ok=True)

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                logging.warning("Empty frame")
                time.sleep(0.1)
                continue

            # Detect
            detections = detector.detect(frame)
            
            # --- Recognition Loop ---
            annotated_frame = frame.copy()
            
            for det in detections:
                box = det["box"]
                kpts = det.get("keypoints")
                
                # Identify
                name, conf = recognizer.identify(frame, kpts=kpts, box=box)
                
                # Logic: Notify or Snapshot
                if name == "Unknown":
                    # Stranger Logic
                    now = time.time()
                    if (now - last_stranger_shot) > stranger_cooldown:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"{stranger_dir}/stranger_{timestamp}.jpg"
                        
                        # Prepare visual alert on the snapshot
                        snap_img = annotated_frame.copy()
                        # Draw the box on the snapshot so we know who it was
                        cv2.rectangle(snap_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                        
                        cv2.imwrite(filename, snap_img)
                        logging.info(f"Stranger detected! Saved snapshot to {filename}")
                        last_stranger_shot = now
                else:
                    # Known Person Logic (Email Alert if needed - implemented in notifier?)
                    # For now just log
                    pass

                # Draw Visuals
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                label = f"{name} ({conf:.2f})"
                
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Draw Keypoints
                if kpts is not None:
                    for kp in kpts:
                        cv2.circle(annotated_frame, (int(kp[0]), int(kp[1])), 2, (0, 255, 255), -1)

            # Show Frame
            cv2.imshow("Security Feed", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        logging.info("Stopping...")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
