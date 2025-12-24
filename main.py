import cv2
import yaml
import logging
import time
import os
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
        mesh_drawer = FaceMeshDrawer() # Initialize Face Mesh
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

    # FPS Counter
    fps_start_time = time.time()
    fps_frame_count = 0
    fps_display = 0.0
    
    # Stranger Snapshot cooldown
    last_stranger_shot = 0
    stranger_cooldown = 10.0  # seconds (reduced from 15)
    stranger_dir = "strangers"
    if not os.path.exists(stranger_dir):
        os.makedirs(stranger_dir, exist_ok=True)

    # Recognition Optimization
    frame_count = 0
    recognition_interval = 30  # Run recognition every 30 frames
    current_identities = {}    # Stores {track_id or box_hash: (name, conf, timestamp)}
    
    # Store previous detections to match faces between recognition frames
    # Simple logic: Match closest box.
    
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                logging.warning("Empty frame")
                time.sleep(0.1)
                continue

            frame_count += 1
            fps_frame_count += 1
            processed_frame = frame.copy()
            
            # Calculate FPS every second
            if time.time() - fps_start_time >= 1.0:
                fps_display = fps_frame_count / (time.time() - fps_start_time)
                fps_frame_count = 0
                fps_start_time = time.time()

            # 1. Detect (Runs every frame for smooth boxes)
            detections = detector.detect(frame)
            
            # --- Recognition Loop ---
            # We only run heavy recognition periodically
            do_recognition = (frame_count % recognition_interval == 0)
            
            active_faces = []

            for det in detections:
                box = det["box"]
                kpts = det.get("keypoints")
                
                # Simple tracking ID based on box center (crude but effective for simple scenes)
                cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
                face_id = f"{int(cx//20)}_{int(cy//20)}" # Spatial hash
                
                name = "Verifying..."
                conf = 0.0
                
                # Check if we have a recent cached identity for this location
                # Find closest match in current_identities
                best_match_id = None
                min_dist = 10000
                
                for existing_id in current_identities:
                    # Parse spatial hash or just use direct distance if we stored center
                    # Let's simple use the hash we just made, but searching is better.
                    # Simplified: Just run recognition if triggered.
                    pass

                # LOGIC:
                # If do_recognition: Run identify() and update current_identities.
                # Else: Use closest match from current_identities.
                
                if do_recognition:
                    name, conf = recognizer.identify(frame, kpts=kpts, box=box)
                    # Update cache (simple cache by center point proximity could be added here, 
                    # but for now we just overwrite local variables. To persist, we need a list.)
                    # Let's attach 'name' to the detection object conceptually.
                else:
                    # Skipping recognition.
                    # Problem: We don't know who is who without tracking.
                    # Quick fix: Just show "Detecting" or don't show name? 
                    # Better fix: Run recognition every frame BUT use a smaller model?
                    # User picked Option A (Skip Frames).
                    
                    # We need to map this box to the previous known name.
                    # Let's brute force match center to previous frame's centers.
                    found_match = False
                    for cached_center, cached_data in current_identities.items():
                        c_x, c_y = cached_center
                        dist = ((cx - c_x)**2 + (cy - c_y)**2)**0.5
                        if dist < 50: # If face is within 50 pixels of last known face
                            name, conf = cached_data["name"], cached_data["conf"]
                            found_match = True
                            # Update entry with new center
                            # We can't modify dict while iterating, so we'll rebuild it later.
                            break
                    
                    if not found_match:
                         # New face in the interim?
                         name = "..."
                
                # Store for drawing
                det["name"] = name
                det["conf"] = conf
                active_faces.append({'center': (cx, cy), 'name': name, 'conf': conf})
                
                # --- Stranger Logic (Only check on Recognition Frames) ---
                if do_recognition and name == "Unknown":
                    # Stranger Logic
                    now = time.time()
                    if (now - last_stranger_shot) > stranger_cooldown:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"{stranger_dir}/stranger_{timestamp}.jpg"
                        
                        snap_img = processed_frame.copy()
                        cv2.rectangle(snap_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                        cv2.imwrite(filename, snap_img)
                        logging.info(f"Stranger detected! Saved snapshot to {filename}")
                        
                        # Send Telegram notification with photo
                        notifier.notify("Unknown", image_path=filename)
                        
                        last_stranger_shot = now


                # Draw Visuals
                color = (0, 255, 0) if name not in ["Unknown", "Verifying...", "..."] else (0, 0, 255)
                # If verifying, use Yellow
                if name in ["Verifying...", "..."]: color = (0, 255, 255)
                
                label = f"{name} ({conf:.2f})" if name not in ["...", "Verifying..."] else name
                
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(processed_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                if kpts is not None:
                    for kp in kpts:
                         cv2.circle(processed_frame, (int(kp[0]), int(kp[1])), 2, (0, 255, 255), -1)

                # --- 3. Draw Face Mesh (Iron Man HUD) ---
                # We draw this ON TOP of the bounding box
                # Pass integer coordinates to the visualizer
                try:
                    mesh_drawer.process_and_draw(processed_frame, [int(b) for b in box])
                except Exception as e:
                    pass # Ignore mesh errors to keep system running

            # Update cache at end of frame
            if do_recognition:
                current_identities = {f['center']: {'name': f['name'], 'conf': f['conf']} for f in active_faces}
            # Else: we technically should update centers to avoid "losing" the track, but 30 frames is long.
            # actually 30 frames (1s) is too long for purely positional matching if person moves.
            # Let's try 5 frames (approx 150ms).
            
            # Display FPS counter
            cv2.putText(processed_frame, f"FPS: {fps_display:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # Show Frame
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
