import cv2
import numpy as np
import logging
from ultralytics import YOLO

class Detector:
    def __init__(self, config):
        self.model_path = config["paths"]["model_path"]
        self.conf_threshold = config["system"]["confidence_threshold"]
        
        # Load Model (Supports .pt and .onnx)
        logging.info(f"Loading Detector model: {self.model_path}")
        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            logging.error(f"Failed to load model {self.model_path}: {e}")
            raise e

    def detect(self, frame):
        """
        Input: frame (BGR image)
        Output: detections list of {"box": [x1, y1, x2, y2], "conf": float, "keypoints": ...}
        """
        detections = []
        
        # Run inference
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
        
        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints if hasattr(result, 'keypoints') else None
            
            for i, box in enumerate(boxes):
                # box.xyxy is [x1, y1, x2, y2]
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Check for 'person' (0) or 'face' (depends on model, usually 0 if single class)
                # If using generic YOLOv8, class 0 is person.
                # If using YOLOv8-Face, class 0 is face.
                # We'll accept class 0.
                if cls != 0:
                    continue

                det = {
                    "box": xyxy,
                    "conf": conf,
                    "label": "face",
                    "keypoints": None
                }
                
                if keypoints is not None and keypoints.xy is not None:
                    # Get keypoints for this detection
                    # shape: (N, 5, 2) or similar
                    if len(keypoints.xy) > i:
                        kpts = keypoints.xy[i].cpu().numpy() # [[x,y], [x,y], ...]
                        det["keypoints"] = kpts

                detections.append(det)
                
        return detections
