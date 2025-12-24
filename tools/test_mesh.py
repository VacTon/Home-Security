import cv2
import mediapipe as mp
import time
import numpy as np

# ==========================================
# CONFIG
# ==========================================
# Only draw these connections for speed & aesthetics
# (Eyes, Eyebrows, Lips, Face Oval)
# We skip the heavy internal mesh lines
MESH_VISUAL_STYLE = mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1)
CONTOUR_STYLE = mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

def run_performance_test():
    # 1. Initialize Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False, # False = 468 points (Faster), True = 478 (Slower)
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 2. Setup Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting Face Mesh Performance Test...")
    print("Strategy: ROI Crop (Only processing the face area)")

    frame_count = 0
    start_time = time.time()
    
    # Fake face box for simulation (normally YOLO provides this)
    # We start with center screen
    face_box = [200, 100, 440, 340] # [x1, y1, x2, y2]
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_h, frame_w, _ = frame.shape
        
        # --- 1. THE OPTIMIZATION: ROI CROP ---
        # Instead of sending the full 640x480 frame to MediaPipe,
        # we only send the face area + some padding.
        
        x1, y1, x2, y2 = face_box
        
        # Add padding (Mesh needs context)
        pad = 20
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(frame_w, x2 + pad)
        y2 = min(frame_h, y2 + pad)
        
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0: continue

        # --- 2. INFERENCE ---
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(roi_rgb)

        # --- 3. DRAWING & TRACKING UPDATE ---
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Update our tracking box based on mesh range
                x_min = min([lm.x for lm in face_landmarks.landmark])
                x_max = max([lm.x for lm in face_landmarks.landmark])
                y_min = min([lm.y for lm in face_landmarks.landmark])
                y_max = max([lm.y for lm in face_landmarks.landmark])
                
                # Convert back to Global Coordinates
                roi_w = x2 - x1
                roi_h = y2 - y1
                
                face_box = [
                    int(x1 + x_min * roi_w), 
                    int(y1 + y_min * roi_h),
                    int(x1 + x_max * roi_w),
                    int(y1 + y_max * roi_h)
                ]
                
                # Draw only on the ROI to save time, or project back to main frame
                # For demo, let's draw strictly the Contours (Faster than mesh)
                mp.solutions.drawing_utils.draw_landmarks(
                    image=roi,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION, # FULL MESH
                    landmark_drawing_spec=None, # Don't draw dots
                    connection_drawing_spec=MESH_VISUAL_STYLE
                )

        # Put ROI back into frame (Visualization only)
        frame[y1:y2, x1:x2] = roi
        
        # --- FPS CALC ---
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed
        
        cv2.rectangle(frame, (face_box[0], face_box[1]), (face_box[2], face_box[3]), (0, 255, 0), 2)
        cv2.putText(frame, f"Mesh FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Face Mesh Efficiency Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_performance_test()
