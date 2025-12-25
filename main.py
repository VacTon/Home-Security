import cv2
import yaml
import logging
import time
import os
import math
import threading
import queue
from camera import Camera
from detector import Detector
from recognizer import Recognizer
from notifier import Notifier
from visualizer import FaceMeshDrawer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

# ==========================================
# THREADED WORKER 
# ==========================================
class AsyncRecognizer:
    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.input_queue = queue.Queue(maxsize=1) 
        self.output_queue = queue.Queue()
        self.last_results = {}
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def submit(self, frame, detections):
        if not self.input_queue.full():
            self.input_queue.put((frame.copy(), detections))

    def get_results(self):
        try:
            while True:
                self.last_results = self.output_queue.get_nowait()
        except queue.Empty:
            pass
        return self.last_results

    def _worker(self):
        while self.running:
            try:
                frame, detections = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            t_start = time.time()
            results = {} 
            for det in detections:
                box = det["box"]
                cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
                name, conf = self.recognizer.identify(frame, kpts=det.get("keypoints"), box=box)
                results[(cx, cy)] = {'name': name, 'conf': conf}
            
            # logging.info(f"Thread Latency: {(time.time()-t_start)*1000:.1f}ms")
            self.output_queue.put(results)

    def stop(self):
        self.running = False


def main():
    config = load_config()
    
    try:
        camera = Camera(width=config["system"]["frame_width"], height=config["system"]["frame_height"])
        detector = Detector(config)
        
        base_recog = Recognizer(config)
        base_recog.load_known_faces()
        recog_worker = AsyncRecognizer(base_recog)
        
        notifier = Notifier(config)
        mesh_drawer = FaceMeshDrawer() 
    except Exception as e:
        logging.error(f"Init failed: {e}")
        return

    try:
        camera.start()
    except RuntimeError:
        return

    logging.info("System started. Threaded Mode + Instrumentation.")

    fps_start_time = time.time()
    fps_frame_count = 0
    fps_display = 0.0
    
    last_stranger_shot = 0
    stranger_cooldown = 10.0
    stranger_dir = "strangers"
    if not os.path.exists(stranger_dir): os.makedirs(stranger_dir, exist_ok=True)

    current_identities = {} 
    
    try:
        frame_idx = 0
        while True:
            t_loop_start = time.time()
            
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Log Frame Size once to verify resolution
            if frame_idx == 0:
                logging.info(f"Video Resolution: {frame.shape}")

            frame_idx += 1
            fps_frame_count += 1
            processed_frame = frame.copy()
            
            if time.time() - fps_start_time >= 1.0:
                fps_display = fps_frame_count / (time.time() - fps_start_time)
                fps_frame_count = 0
                fps_start_time = time.time()

            # TIMING SECTION
            t0 = time.time()
            
            # 1. Detect
            detections = detector.detect(frame)
            t1 = time.time()
            
            # 2. Async Recog
            recog_worker.submit(frame, detections)
            new_results = recog_worker.get_results()
            t2 = time.time()
            
            # 3. Tracking & Logic
            tracked_faces = []
            for det in detections:
                box = det["box"]
                cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
                name = "Verifying..."
                conf = 0.0
                
                matched = False
                if new_results:
                    best = None
                    mn = 100.0
                    for rc, rd in new_results.items():
                        d = math.hypot(cx - rc[0], cy - rc[1])
                        if d < mn:
                            mn = d
                            best = rd
                    if best:
                        name, conf = best['name'], best['conf']
                        matched = True
                
                if not matched:
                    best = None
                    mn = 100.0
                    for kc, kd in current_identities.items():
                         d = math.hypot(cx - kc[0], cy - kc[1])
                         if d < mn:
                             mn = d
                             best = kd
                    if best:
                        name, conf = best['name'], best['conf']

                tracked_faces.append({'center': (cx, cy), 'name': name, 'conf': conf, 'box': box})
            
            # === DEDUPLICATION: Ensure no two faces have the same identity ===
            # Build a map of name -> list of faces claiming that name
            name_claims = {}
            for face in tracked_faces:
                face_name = face['name']
                if face_name not in ["Unknown", "Verifying...", "..."]:
                    if face_name not in name_claims:
                        name_claims[face_name] = []
                    name_claims[face_name].append(face)
            
            # Resolve conflicts: keep highest confidence, demote others
            for claimed_name, claimants in name_claims.items():
                if len(claimants) > 1:
                    # Sort by confidence (descending)
                    claimants.sort(key=lambda f: f['conf'], reverse=True)
                    
                    # Winner keeps the name
                    winner = claimants[0]
                    
                    # Losers become Unknown
                    for loser in claimants[1:]:
                        loser['name'] = "Unknown"
                        loser['conf'] = 0.0
                        logging.info(f"Deduplication: Demoted duplicate '{claimed_name}' (conf={loser['conf']:.2f}) to Unknown")
            
            # === DRAWING & ALERTS (using deduplicated data) ===
            for face in tracked_faces:
                name = face['name']
                conf = face['conf']
                box = face['box']
                
                # Drawing
                color = (0, 255, 0)
                if name == "Unknown": color = (0, 0, 255)
                if name == "Verifying...": color = (255, 255, 0)

                label = f"{name} ({conf:.2f})"
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(processed_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Face Mesh
                try:
                    mesh_drawer.process_and_draw(processed_frame, [int(b) for b in box])
                except Exception as e:
                    pass

                # Stranger Alert
                if name == "Unknown" and (time.time() - last_stranger_shot) > stranger_cooldown:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"{stranger_dir}/stranger_{timestamp}.jpg"
                    snap = frame.copy()
                    cv2.rectangle(snap, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.imwrite(filename, snap)
                    notifier.notify("Unknown", image_path=filename)
                    last_stranger_shot = time.time()

            
            t3 = time.time()
            
            # Commit Updates
            current_identities = {f['center']: {'name': f['name'], 'conf': f['conf']} for f in tracked_faces}
            
            cv2.putText(processed_frame, f"FPS: {fps_display:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("Security Feed", processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Performance Log
            if frame_idx % 30 == 0:
                logging.info(f"Times (ms) | Det: {(t1-t0)*1000:.1f} | Async+Track: {(t2-t1)*1000:.1f} | Draw: {(t3-t2)*1000:.1f}")

    except KeyboardInterrupt:
        pass
    finally:
        recog_worker.stop()
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
