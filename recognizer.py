import cv2
import numpy as np
import logging
import os
import pickle
from hailo_platform import VDevice, HEF, ConfigureParams, InferVStreams, InputVStreamParams, OutputVStreamParams

class Recognizer:
    def __init__(self, config):
        self.faces_dir = config["paths"]["faces_dir"]
        self.model_path = config["paths"].get("recognition_model_path", "models/arcface_mobilefacenet.hef")
        self.tolerance = config["system"]["recognition_tolerance"]
        
        self.known_encodings = []
        self.known_names = []
        
        # 0. Initialize placeholders to avoid AttributeErrors on failure
        self.vdevice = None
        self.network_group = None
        self.output_vstream_infos = []
        
        # 1. Initialize Hailo Device
        logging.info(f"Initializing Hailo Device for Recognition: {self.model_path}")
        try:
            self.vdevice = VDevice()
            self.hef = HEF(self.model_path)
            
            # Use the newer API method to create configure params
            self.configure_params = self.hef.create_configure_params()
            self.network_group = self.vdevice.configure(self.hef, self.configure_params)[0]
            
            # Get stream info
            self.input_vstream_infos = self.hef.get_input_vstream_infos()
            self.output_vstream_infos = self.hef.get_output_vstream_infos()
            
            self.input_name = self.input_vstream_infos[0].name
            self.output_name = self.output_vstream_infos[0].name
            
            # Input shape (H, W, C)
            self.input_shape = self.input_vstream_infos[0].shape
            logging.info(f"Hailo Model Loaded. Input: {self.input_name} {self.input_shape}, Output: {self.output_name}")
            
        except Exception as e:
            logging.error(f"Failed to initialize Hailo Recognition: {e}")
            # Keep self.vdevice as None to trigger fallback if any

        # Standard ArcFace 5-point landmarks (for 112x112)
        self.target_kps = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)

    def preprocess(self, img, kpts=None, box=None):
        """Aligns and prepares face for Hailo."""
        # 1. Align
        if kpts is not None and len(kpts) == 5:
            st = cv2.estimateAffinePartial2D(kpts, self.target_kps, method=cv2.LMEDS)[0]
            face_img = cv2.warpAffine(img, st, (112, 112), borderValue=0.0)
        elif box is not None:
            x1, y1, x2, y2 = map(int, box)
            face_img_raw = img[max(0, y1):y2, max(0, x1):x2]
            if face_img_raw.size == 0: return None
            face_img = cv2.resize(face_img_raw, (112, 112))
        else:
            return None

        # 2. Format for Hailo
        # Most Hailo HEFs for ArcFace expect RGB, uint8
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        # Note: If the HEF was compiled with normalization, we don't divide by 255.
        # MobileFaceNet HEF usually takes 0-255 uint8.
        return face_img.astype(np.uint8)

    def get_embedding(self, img, kpts=None, box=None):
        if self.vdevice is None:
            return None
        
        face_blob = self.preprocess(img, kpts, box)
        if face_blob is None:
            return None
            
        # 3. Hailo Inference
        input_data = {self.input_name: np.expand_dims(face_blob, axis=0)}
        
        # Use simple inference pipeline
        input_params = InputVStreamParams.make_from_network_group(self.network_group, quantized=False)
        output_params = OutputVStreamParams.make_from_network_group(self.network_group, quantized=False)
        
        with InferVStreams(self.network_group, input_params, output_params) as infer_pipeline:
            # Activate and run
            with self.network_group.activate_config(self.configure_params):
                results = infer_pipeline.infer(input_data)
                emb = results[self.output_name][0] # Get first batch
                
        # 4. Normalize
        norm = np.linalg.norm(emb)
        return (emb / (norm + 1e-10)).reshape(1, -1)

    def load_known_faces(self):
        logging.info("Loading known faces...")
        encodings_file = os.path.join(self.faces_dir, "encodings.pkl")
        
        if os.path.exists(encodings_file):
             with open(encodings_file, "rb") as f:
                 data = pickle.load(f)
                 # Ensure we stack into a single matrix (N, Dim)
                 self.known_encodings = np.vstack(data["encodings"]) if data["encodings"] else np.array([])
                 self.known_names = data["names"]
                 
                 logging.info(f"Loaded {len(self.known_names)} faces from cache.")
                 # Check dimension. Old ArcFace R50 might be 512, MobileFaceNet might be 128.
                 if len(self.known_encodings) > 0 and self.known_encodings.shape[1] != self.output_vstream_infos[0].shape[0]:
                     logging.warning("!!! CACHE DIMENSION MISMATCH !!!")
                     logging.warning(f"Cache has {self.known_encodings.shape[1]}d, Model needs {self.output_vstream_infos[0].shape[0]}d.")
                     logging.warning("You MUST re-run 'python tools/process_database.py' to update your faces!")
                 return

        logging.info("No cache found. Please run 'python tools/process_database.py' first.")

    def identify(self, frame, kpts=None, box=None):
        if not self.known_encodings:
            return "Unknown", 0.0

        target_emb = self.get_embedding(frame, kpts, box) 
        if target_emb is None:
            return "Unknown", 0.0
            
        # Cosine Similarity
        sims = np.dot(self.known_encodings, target_emb.T).flatten()
        best_idx = np.argmax(sims)
        best_sim = sims[best_idx]
        
        if best_sim > self.tolerance:
             return self.known_names[best_idx], float(best_sim)
             
        return "Unknown", float(best_sim)

    def __del__(self):
        # Cleanup device
        if hasattr(self, 'vdevice') and self.vdevice:
            # VDevice doesn't always need explicit close in some python wrappers,
            # but it is good practice.
            pass
