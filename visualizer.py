import cv2
import mediapipe as mp
import numpy as np

class FaceMeshDrawer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        # Use the "Lightning" fast settings
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Custom "Cyberpunk" Style
        # Teal connections, Green dots
        self.connection_style = mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=1)
        self.landmark_style = mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1)

    def process_and_draw(self, frame, face_box):
        """
        Runs Face Mesh on the cropped face_box and draws it back onto the main frame.
        """
        try:
            h, w, c = frame.shape
            x1, y1, x2, y2 = face_box
            
            # 1. ADD PADDING
            # Context helps the mesh model find the chin/forehead
            pad = 20
            px1 = max(0, x1 - pad)
            py1 = max(0, y1 - pad)
            px2 = min(w, x2 + pad)
            py2 = min(h, y2 + pad)
            
            roi_w = px2 - px1
            roi_h = py2 - py1
            
            if roi_w < 10 or roi_h < 10: return

            # 2. CROP
            roi = frame[py1:py2, px1:px2]
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
            # 3. RUN MESH
            results = self.face_mesh.process(roi_rgb)
            
            # 4. DRAW (Projecting back to global coordinates)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # We manually draw to handle the coordinate offset
                    # (Standard mediapipe utils can't map ROI -> Global easily)
                    
                    # Optimization: Only draw connections (lines), skip dots
                    mp.solutions.drawing_utils.draw_landmarks(
                        image=frame[py1:py2, px1:px2], # Draw directly on the ROI slice of the main frame
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION, 
                        landmark_drawing_spec=None, # Skip dots for speed
                        connection_drawing_spec=self.connection_style
                    )
                    
                    # Draw a cool "Center Lock" circle on the nose tip (Index 1)
                    nose_tip = face_landmarks.landmark[1]
                    nx = int(px1 + nose_tip.x * roi_w)
                    ny = int(py1 + nose_tip.y * roi_h)
                    
                    # Draw futuristic crosshair
                    cv2.circle(frame, (nx, ny), 4, (0, 0, 255), -1)
                    cv2.line(frame, (nx-10, ny), (nx+10, ny), (0, 0, 255), 1)
                    cv2.line(frame, (nx, ny-10), (nx, ny+10), (0, 0, 255), 1)

        except Exception as e:
            # Don't crash the whole security system if mesh fails
            pass
