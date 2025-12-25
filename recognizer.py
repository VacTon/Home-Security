import cv2
import numpy as np
import logging
import os
import pickle
import onnxruntime as ort

class Recognizer:
    def __init__(self, config):
        self.faces_dir = config["paths"]["faces_dir"]
        # Use ONNX model for CPU inference (reliable and fast enough)
        self.model_path = config["paths"].get("recognition_model_path", "models/w600k_r50.onnx")
        self.tolerance = config["system"]["recognition_tolerance"]
        
        self.known_encodings = []
        self.known_names = []
        
        # Initialize ONNX Runtime (CPU)
        logging.info(f"Initializing Face Recognition (ONNX/CPU): {self.model_path}")
        try:
            self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            input_shape = self.session.get_inputs()[0].shape
            logging.info(f"Recognition Model Loaded. Input: {input_shape}")
        except Exception as e:
            logging.error(f"Failed to load recognition model: {e}")
            self.session = None

        # Standard ArcFace 5-point landmarks (for 112x112)
        self.target_kps = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)

    def preprocess(self, img, kpts=None, box=None):
        """Aligns and prepares face for recognition."""
        # 1. Align face
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

        # 2. Normalize for ONNX model
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = face_img.astype(np.float32) / 255.0
        face_img = (face_img - 0.5) / 0.5  # Normalize to [-1, 1]
        face_img = np.transpose(face_img, (2, 0, 1))  # HWC -> CHW
        return face_img

    def get_embedding(self, img, kpts=None, box=None):
        if self.session is None:
            return None
        
        face_blob = self.preprocess(img, kpts, box)
        if face_blob is None:
            return None
            
        # Run ONNX inference
        try:
            input_data = np.expand_dims(face_blob, axis=0).astype(np.float32)
            outputs = self.session.run([self.output_name], {self.input_name: input_data})
            emb = outputs[0][0]
            
            # Normalize
            norm = np.linalg.norm(emb)
            return (emb / (norm + 1e-10)).reshape(1, -1)
            
        except Exception as e:
            logging.error(f"Inference failed: {e}")
            return None

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
                 
                 # Check dimension (ONNX model outputs 512-d embeddings)
                 if self.session and len(self.known_encodings) > 0:
                     expected_dim = self.session.get_outputs()[0].shape[1]
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


