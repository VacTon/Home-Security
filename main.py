import cv2
import yaml
import logging
import time
import os
import math
import multiprocessing as mp
import queue
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

# ==========================================
# MULTIPROCESSING WORKER FUNCTION
# ==========================================
def recognition_process_func(input_q, output_q, config):
    """
    Runs in a separate CPU PROCESS.
    Has its own memory space and bypasses Python GIL.
    """
    try:
        logging.info("Recognizer Process Starting...")
        # Re-initialize modules inside the process process
        # (ONNX sessions cannot be shared across processes)
        recognizer = Recognizer(config)
        recognizer.load_known_faces()
        logging.info("Recognizer Process Ready.")
        
        while True:
            try:
                frame, detections = input_q.get(timeout=1.0)
            except queue.Empty:
                continue
            
            if frame is None: # Sentinel to exit
                break

            results = {} # {(cx,cy) -> {'name': name, 'conf': conf}}
            
            for det in detections:
                box = det["box"]
                cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
                
                # Run Recognition (Heavy CPU Op)
                name, conf = recognizer.identify(frame, kpts=det.get("keypoints"), box=box)
                
                results[(cx, cy)] = {'name': name, 'conf': conf}
            
            output_q.put(results)
            
    except Exception as e:
        logging.error(f"Process Crash: {e}")

# ==========================================
# ASYNC WRAPPER
# ==========================================
class AsyncRecognizer:
    def __init__(self, config):
        self.input_queue = mp.Queue(maxsize=1) 
        self.output_queue = mp.Queue()
        
        # Start the heavy process
        self.process_worker = mp.Process(target=recognition_process_func, 
                                        args=(self.input_queue, self.output_queue, config))
        self.process_worker.daemon = True
        self.process_worker.start()

    def process(self, frame, detections):
        if self.input_queue.full():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                pass
        self.input_queue.put((frame.copy(), detections))

    def get_latest_results(self):
        latest = None
        try:
            while True:
                latest = self.output_queue.get_nowait()
        except queue.Empty:
            pass
        return latest

    def stop(self):
        self.input_queue.put((None, None))
        self.process_worker.join(timeout=1.0)

# ==========================================
# MAIN LOOP
# ==========================================
def main():
    config = load_config()
    
    # Initialize Modules (Main Thread)
    try:
        # Camera & Detector (Lightweight Logic)
        camera = Camera(width=config["system"]["frame_width"], height=config["system"]["frame_height"])
        detector = Detector(config) # YOLO is fairly fast on CPU
        notifier = Notifier(config)
        mesh_drawer = FaceMeshDrawer() 
        
        # Async Recognizer (Heavy Logic - Separate Core)
        recog_worker = AsyncRecognizer(config)
        
    except Exception as e:
        logging.error(f"Initialization failed: {e}")
        return

    # Start Camera
    try:
        camera.start()
    except RuntimeError:
        logging.error("Failed to start camera.")
        return

    logging.info("System started. Press 'q' to quit.")

    fps_start_time = time.time()
    fps_frame_count = 0
    fps_display = 0.0
    
    last_stranger_shot = 0
    stranger_cooldown = 10.0
    stranger_dir = "strangers"
    if not os.path.exists(stranger_dir): os.makedirs(stranger_dir, exist_ok=True)

    current_identities = {} 
    
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            fps_frame_count += 1
            processed_frame = frame.copy()
            
            if time.time() - fps_start_time >= 1.0:
                fps_display = fps_frame_count / (time.time() - fps_start_time)
                fps_frame_count = 0
                fps_start_time = time.time()

            # 1. Detect (Main Process)
            detections = detector.detect(frame)
            
            # 2. Submit to Background Core
            recog_worker.process(frame, detections)
            
            # 3. Retrieve Background Core Results
            new_results = recog_worker.get_latest_results()
            
            # 4. TRACKING LOGIC (Match Stale Results to Fresh Detections)
            tracked_faces = []

            for det in detections:
                box = det["box"]
                cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
                
                name = "Verifying..."
                conf = 0.0
                
                # A. Try Stale AI Result First
                matched_ai = False
                if new_results:
                    best_match = None
                    min_dist = 100.0
                    for res_center, res_data in new_results.items():
                        dist = math.hypot(cx - res_center[0], cy - res_center[1])
                        if dist < min_dist:
                            min_dist = dist
                            best_match = res_data
                    
                    if best_match:
                        name = best_match['name']
                        conf = best_match['conf']
                        matched_ai = True
                
                # B. Fallback to Memory
                if not matched_ai:
                    best_match = None
                    min_dist = 100.0
                    for known_center, known_data in current_identities.items():
                         dist = math.hypot(cx - known_center[0], cy - known_center[1])
                         if dist < min_dist:
                             min_dist = dist
                             best_match = known_data
                    
                    if best_match:
                        name = best_match['name']
                        conf = best_match['conf']

                tracked_faces.append({'center': (cx, cy), 'name': name, 'conf': conf})
                
                # Drawing
                color = (0, 255, 0)
                if name == "Unknown": color = (0, 0, 255)
                if name in ["Verifying...", "..."]: color = (255, 255, 0)

                label = f"{name} ({conf:.2f})"
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(processed_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                try: # Face Mesh Overlay
                    mesh_drawer.process_and_draw(processed_frame, [int(b) for b in box])
                except:
                    pass

                # Stranger Alert
                if name == "Unknown":
                     now = time.time()
                     if (now - last_stranger_shot) > stranger_cooldown:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"{stranger_dir}/stranger_{timestamp}.jpg"
                        snap = frame.copy()
                        cv2.rectangle(snap, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.imwrite(filename, snap)
                        notifier.notify("Unknown", image_path=filename)
                        last_stranger_shot = now

            # Commit Updates
            current_identities = {f['center']: {'name': f['name'], 'conf': f['conf']} for f in tracked_faces}
            
            cv2.putText(processed_frame, f"FPS: {fps_display:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("Security Feed", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        logging.info("Stopping...")
    finally:
        recog_worker.stop()
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
