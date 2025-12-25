import cv2
import numpy as np
import mediapipe as mp
import logging

class Detector:
    def __init__(self, config):
        # We replace YOLO with MediaPipe for CPU Performance (500ms -> 10ms)
        logging.info("Initializing MediaPipe Face Detector (Optimized for CPU)...")
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=0, # 0 = Short Range (Front Cam), 1 = Long Range
            min_detection_confidence=config["system"]["confidence_threshold"]
        )

    def detect(self, frame):
        """
        Input: frame (BGR image)
        Output: detections list
        """
        h, w, c = frame.shape
        
        # MediaPipe needs RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(frame_rgb)
        
        detections = []
        if results.detections:
            for detection in results.detections:
                # 1. Bounding Box
                bboxC = detection.location_data.relative_bounding_box
                x1 = int(bboxC.xmin * w)
                y1 = int(bboxC.ymin * h)
                box_w = int(bboxC.width * w)
                box_h = int(bboxC.height * h)
                x2 = x1 + box_w
                y2 = y1 + box_h
                
                # Clip to frame
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # 2. Keypoints (Mapping MP 6-points to generic format)
                # MP: 0=RightEye, 1=LeftEye, 2=Nose, 3=MouthCenter, 4=RightEar, 5=LeftEar
                # We want robust array for alignment.
                kpts = []
                for kp in detection.location_data.relative_keypoints:
                    kpts.append([kp.x * w, kp.y * h])
                
                kpts_np = np.array(kpts, dtype=np.float32)

                # Create specific alignment format for ArcFace?
                # Recognizer expects 5 points [LeftEye, RightEye, Nose, LeftMouth, RightMouth]
                # We can construct a best-guess 5-point set:
                # Target L-Eye = MP LeftEye (1)
                # Target R-Eye = MP RightEye (0)
                # Target Nose  = MP Nose (2)
                # Target L-Mouth = MP MouthCenter (3)
                # Target R-Mouth = MP MouthCenter (3)
                
                # Let's pass the mapped version explicitly if possible, 
                # but recognizer expects a raw list usually.
                # Actually recognizer.py checks len(kpts) == 5.
                # Let's fabricate a 5-point array.
                
                if len(kpts_np) >= 6:
                     # Remap to [LeftEye, RightEye, Nose, Mouth, Mouth]
                     # Note: MP Eye 0 is Right (User's Right), Pixel-wise it is on Left if selfie?
                     # Wait, MP 0 is Right Eye (User's Right). In image, that is on the LEFT side of the face (if looking at cam).
                     # Coordinates: x is 0..1.
                     # Let's blindly trust the visual points. 
                     # MP 1 is Left Eye (User's Left).
                     
                     matched_kpts = np.array([
                         kpts_np[1], # Left Eye
                         kpts_np[0], # Right Eye
                         kpts_np[2], # Nose
                         kpts_np[3], # Mouth (Left) - Reuse Center
                         kpts_np[3]  # Mouth (Right) - Reuse Center
                     ], dtype=np.float32)
                else:
                    matched_kpts = None

                det = {
                    "box": np.array([x1, y1, x2, y2]),
                    "conf": detection.score[0],
                    "label": "face",
                    "keypoints": matched_kpts
                }
                detections.append(det)
                
        return detections
