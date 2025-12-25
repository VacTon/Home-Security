import cv2
import yaml
import logging
import time
import os
import math
from camera import Camera
from detector import Detector
from recognizer import Recognizer
from notifier import Notifier
from visualizer import FaceMeshDrawer

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
        mesh_drawer = FaceMeshDrawer() 
    except Exception as e:
        logging.error(f"Initialization failed: {e}")
        return

    # Load known faces from cache
    recognizer.load_known_faces()

    # Start Camera
    try:
        camera.start()
    except RuntimeError:
        logging.error("Failed to start camera.")
        return

    logging.info("System started. Press 'q' to quit.")

    # FPS Counter
    fps_start_time = time.time()
    fps_frame_count = 0
    fps_display = 0.0
    
    # Stranger Snapshot cooldown
    last_stranger_shot = 0
    stranger_cooldown = 10.0
    stranger_dir = "strangers"
    if not os.path.exists(stranger_dir):
        os.makedirs(stranger_dir, exist_ok=True)

    # Recognition Control
    frame_count = 0
    recognition_interval = 5  # Run recognition every 5 frames (~6 times/sec) for better tracking
    current_identities = {}   # Mapping: (cx, cy) -> {'name': str, 'conf': float}
    
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            fps_frame_count += 1
            processed_frame = frame.copy()
            
            # FPS Calculation
            if time.time() - fps_start_time >= 1.0:
                fps_display = fps_frame_count / (time.time() - fps_start_time)
                fps_frame_count = 0
                fps_start_time = time.time()

            # 1. Detect (YOLOv8)
            detections = detector.detect(frame)
            
            # 2. Recognition & Tracking Strategy
            do_recognition = (frame_count % recognition_interval == 0)
            
            # We build a new registry of identities based on CURRENT positions
            next_frame_identities = {}
            
            for det in detections:
                box = det["box"]
                cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
                
                name = "..."
                conf = 0.0
                
                if do_recognition:
                    # Run Deep Learning Recognition
                    name, conf = recognizer.identify(frame, kpts=det.get("keypoints"), box=box)
                else:
                    # Run Positional Tracking (Soft Match)
                    # Look for the closest face from the PREVIOUS frame
                    best_match = None
                    min_dist = 100.0 # Pixel distance threshold for tracking
                    
                    for known_center, known_data in current_identities.items():
                        # Euclidean Distance
                        dist = math.hypot(cx - known_center[0], cy - known_center[1])
                        if dist < min_dist:
                            min_dist = dist
                            best_match = known_data
                    
                    if best_match:
                        name = best_match['name']
                        conf = best_match['conf']
                
                # Update Tracker for next frame
                next_frame_identities[(cx, cy)] = {'name': name, 'conf': conf}
                
                # Stranger Alert Logic (Only on recognition frames to avoid spam)
                if do_recognition and name == "Unknown":
                    now = time.time()
                    if (now - last_stranger_shot) > stranger_cooldown:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"{stranger_dir}/stranger_{timestamp}.jpg"
                        
                        snap_img = processed_frame.copy()
                        cv2.rectangle(snap_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                        cv2.imwrite(filename, snap_img)
                        logging.info(f"Stranger detected! Saved: {filename}")
                        notifier.notify("Unknown", image_path=filename)
                        last_stranger_shot = now

                # Drawing
                color = (0, 255, 0) # Green for Known
                if name == "Unknown": color = (0, 0, 255) # Red for Stranger
                if name == "...": color = (255, 255, 0) # Cyan/Yellow for Processing
                
                label = f"{name} ({conf:.2f})"
                x1, y1, x2, y2 = map(int, box)
                
                # Draw Box & Label
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(processed_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Draw Face Mesh (Overlay)
                try:
                    mesh_drawer.process_and_draw(processed_frame, [int(b) for b in box])
                except Exception:
                    pass 

            # Commit Tracking Updates
            current_identities = next_frame_identities
            
            # FPS Overlay
            cv2.putText(processed_frame, f"FPS: {fps_display:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            cv2.imshow("Security Feed", processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        logging.info("Stopping...")
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
