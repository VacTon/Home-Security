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
        
        # 0. Initialize placeholders
        self.vdevice = None
        self.infer_model = None
        
        # 1. Initialize Hailo Device (Modern High-Level API)
        logging.info(f"Initializing Hailo Recognition (High-Level API): {self.model_path}")
        try:
            self.vdevice = VDevice()
            self.infer_model = self.vdevice.create_infer_model(self.model_path)
            
            # Identify input/output names
            self.input_name = self.infer_model.input().name
            self.output_name = self.infer_model.output().name
            
            # Set input shape (H, W, C)
            self.input_shape = self.infer_model.input().shape
            logging.info(f"Hailo Model Loaded. Input: {self.input_name} {self.input_shape}")
            
        except Exception as e:
            logging.error(f"Failed to initialize Hailo Recognition: {e}")

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
        if self.infer_model is None:
            return None
        
        face_blob = self.preprocess(img, kpts, box)
        if face_blob is None:
            return None
            
        # 3. Hailo Inference (High-Level API with Bindings)
        try:
            with self.infer_model.configure() as configured_model:
                bindings = configured_model.create_bindings()
                
                # Prepare Output Buffer
                output_shape = self.infer_model.output().shape
                # Usually MobileFaceNet/ArcFace returns [512] or [128]
                output_buffer = np.empty(output_shape, dtype=np.float32)
                
                # Bind Input
                # Note: Some models expect batch dim, some don't. 
                # We'll try to match the expected shape.
                input_data = face_blob
                if len(self.infer_model.input().shape) == 4:
                     input_data = np.expand_dims(face_blob, axis=0)
                
                bindings.input(self.input_name).set_buffer(input_data)
                bindings.output(self.output_name).set_buffer(output_buffer)
                
                # Execute (Pass bindings as a LIST)
                configured_model.run([bindings], 1000) # 1000ms timeout
                emb = output_buffer.copy()
                
        except Exception as e:
            logging.error(f"Inference failed: {e}")
            return None
                
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
                 
                 # Check dimension.
                 if self.infer_model and len(self.known_encodings) > 0:
                     expected_dim = self.infer_model.output().shape[0]
                     if self.known_encodings.shape[1] != expected_dim:
                         logging.warning("!!! CACHE DIMENSION MISMATCH !!!")
                         logging.warning(f"Cache: {self.known_encodings.shape[1]}d, Model: {expected_dim}d.")
                         logging.warning("Please re-run 'python tools/process_database.py'")
                 return

        logging.info("No cache found. Please run 'python tools/process_database.py' first.")

    def identify(self, frame, kpts=None, box=None):
        if self.known_encodings.size == 0:
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
