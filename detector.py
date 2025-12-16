import cv2
import numpy as np
import logging

class Detector:
    def __init__(self, config):
        self.model_path = config["paths"]["model_path"]
        self.conf_threshold = config["system"]["confidence_threshold"]
        self.width = config["system"]["frame_width"]
        self.height = config["system"]["frame_height"]
        
        # This will need the HailoRT Python API installed on the Pi
        # For development/testing on PC, we might fail to load this.
        self.use_hailo = False
        try:
            from hailo_platform import VDevice, HEF, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType
            self.use_hailo = True
            logging.info("Hailo Platform library found. Using Hailo hardware.")
        except ImportError:
            logging.warning("Hailo Platform library NOT found. Falling back to CPU/Ultralytics (if installed) or Dummy.")
            self.use_hailo = False
            
        self.hef = None
        self.network_group = None
        self.network_group_params = None
        self.input_vstream_params = None
        self.output_vstream_params = None
        
        if self.use_hailo:
            self._init_hailo()
        else:
            self._init_fallback()

    def _init_hailo(self):
        # Placeholder for full Hailo initialization
        # This requires opening the VDevice, loading HEF, and configuring vstreams
        # Since this is complex and hardware-dependent, we'll sketch the structure.
        # meaningful implementation requires the 'hailo_platform' objects.
        pass

    def _init_fallback(self):
        try:
            from ultralytics import YOLO
            logging.info("Loading YOLOv8n (CPU) for fallback...")
            self.fallback_model = YOLO("yolov8n.pt") # Will download if missing
            self.fallback_model.fuse()
        except ImportError:
            self.fallback_model = None
            logging.warning("Ultralytics not installed. Detection will return empty.")

    def detect(self, frame):
        """
        Input: frame (BGR image)
        Output: detections list of {"box": [x1, y1, x2, y2], "conf": float, "label": str}
        """
        detections = []
        
        # 1. Fallback / PC Mode
        if not self.use_hailo and self.fallback_model:
            results = self.fallback_model.predict(frame, conf=self.conf_threshold, verbose=False)
            for box in results[0].boxes:
                # box.xyxy is [x1, y1, x2, y2]
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.fallback_model.names[cls]
                
                # Filter for 'person' class only (class 0 in COCO)
                if cls == 0: 
                    detections.append({
                        "box": xyxy,
                        "conf": conf,
                        "label": "person"
                    })
            return detections
            
        # 2. Hailo Mode (Skeleton)
        if self.use_hailo:
            # Logic:
            # 1. Resize frame to model input
            # 2. Send to input vstream
            # 3. Read from output vstream
            # 4. Post-process (decode boxes)
            # This part is highly specific to the trained model's output layers.
            pass
            
        return detections
