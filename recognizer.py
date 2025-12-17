import cv2
import numpy as np
import logging
import os
import onnxruntime as ort
import pickle

class Recognizer:
    def __init__(self, config):
        self.faces_dir = config["paths"]["faces_dir"]
        self.model_path = config["paths"].get("recognition_model_path", "models/w600k_r50.onnx")
        self.tolerance = config["system"]["recognition_tolerance"] # e.g. 0.6 for ArcFace (cosine distance)
        
        self.known_encodings = []
        self.known_names = []
        
        # Load ONNX Model
        if os.path.exists(self.model_path):
            logging.info(f"Loading Recognition Model: {self.model_path}")
            try:
                providers = ['CPUExecutionProvider'] # Add 'CUDAExecutionProvider' if on PC/Jetson
                self.session = ort.InferenceSession(self.model_path, providers=providers)
                self.input_name = self.session.get_inputs()[0].name
                self.input_shape = self.session.get_inputs()[0].shape # [1, 3, 112, 112] usually
            except Exception as e:
                logging.error(f"Failed to load recognition model: {e}")
                self.session = None
        else:
            logging.warning(f"Recognition model not found at {self.model_path}. Recognition will fail.")
            self.session = None

        # Standard ArcFace 5-point landmarks (112x112)
        self.target_kps = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)

    def preprocess(self, img, kpts=None, box=None):
        """
        Aligns and normalizes face.
        Input: 
            img: Full frame
            kpts: [5, 2] points (eye_l, eye_r, nose, mouth_l, mouth_r)
            box: [x1, y1, x2, y2] fallback if kpts missing
        Output:
            aligned_face: (1, 3, 112, 112) normalized tensor
        """
        # 1. Align
        if kpts is not None and kpts.shape == (5, 2):
            st = cv2.estimateAffinePartial2D(kpts, self.target_kps, method=cv2.LMEDS)[0]
            face_img = cv2.warpAffine(img, st, (112, 112), borderValue=0.0)
        else:
            # Fallback: Crop and Resize
            x1, y1, x2, y2 = box
            w, h = x2-x1, y2-y1
            # Add some margin?
            face_img_raw = img[max(0, y1):y2, max(0, x1):x2]
            if face_img_raw.size == 0:
                return None
            face_img = cv2.resize(face_img_raw, (112, 112))

        # 2. Normalize (0-255 -> -1 to 1 or 0-1 depending on model)
        # Standard ArcFace is (pixel - 127.5) / 128.0
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = np.transpose(face_img, (2, 0, 1)) # HWC -> CHW
        face_img = np.expand_dims(face_img, axis=0).astype(np.float32)
        face_img = (face_img - 127.5) / 128.0
        
        return face_img

    def get_embedding(self, img, kpts=None, box=None):
        if self.session is None:
            return None
        
        blob = self.preprocess(img, kpts, box)
        if blob is None:
            return None
            
        emb = self.session.run(None, {self.input_name: blob})[0]
        # Normalize embedding to unit length
        # emb shape (1, 512)
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / (norm + 1e-10)

    def load_known_faces(self):
        logging.info("Loading known faces...")
        if not os.path.exists(self.faces_dir):
            os.makedirs(self.faces_dir, exist_ok=True)
            return

        # Look for pre-saved encodings
        encodings_file = os.path.join(self.faces_dir, "encodings.pkl")
        if os.path.exists(encodings_file):
             with open(encodings_file, "rb") as f:
                 data = pickle.load(f)
                 self.known_encodings = data["encodings"]
                 self.known_names = data["names"]
                 logging.info(f"Loaded {len(self.known_names)} faces from cache.")
                 return # We can skip rescanning or merge. For now return.

        # Scan directories
        for person_name in os.listdir(self.faces_dir):
            person_dir = os.path.join(self.faces_dir, person_name)
            if not os.path.isdir(person_dir):
                continue
                
            for filename in os.listdir(person_dir):
                filepath = os.path.join(person_dir, filename)
                img = cv2.imread(filepath)
                if img is None:
                    continue
                
                # Assume image contains only the face or main face
                # We need to detect keypoints for best alignment.
                # However, our detector is separate. 
                # For simplicity in 'loading', we might just resize if no external detector is used here.
                # Ideally, we should use the Detector inside here too, but circular dependency.
                # Simple Hack: Resize to 112x112 directly if it looks like a crop, 
                # Or simplistic center crop.
                
                # BETTER: Use a separate alignment step for database creation.
                # For now, we'll try just resizing/cropping center.
                h, w, _ = img.shape
                # Center crop
                # ...
                # Actually, let's just resize. It reduces accuracy but works for a prototype.
                blob = cv2.resize(img, (112, 112))
                # Normalize logic same as convert
                blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
                blob = np.transpose(blob, (2, 0, 1))
                blob = np.expand_dims(blob, axis=0).astype(np.float32)
                blob = (blob - 127.5) / 128.0
                
                if self.session:
                    emb = self.session.run(None, {self.input_name: blob})[0]
                    norm = np.linalg.norm(emb, axis=1, keepdims=True)
                    emb = emb / norm
                    
                    self.known_encodings.append(emb[0])
                    self.known_names.append(person_name)

        logging.info(f"Loaded {len(self.known_names)} known faces.")

    def identify(self, frame, kpts=None, box=None):
        """
        Returns: name (str), confidence (float)
        """
        if not self.known_encodings:
            return "Unknown", 0.0

        target_emb = self.get_embedding(frame, kpts, box) 
        if target_emb is None:
            return "Unknown", 0.0
            
        # Compare with known
        # Cosine Similarity = dot product (since vectors are unit length)
        # target_emb is (1, 512)
        # known is (N, 512)
        sims = np.dot(self.known_encodings, target_emb.T).flatten() # (N,)
        
        best_idx = np.argmax(sims)
        best_sim = sims[best_idx]
        
        # ArcFace thresholds usually: 0.25 (strict) to 0.4 (loose)?? 
        # Actually cosine sim: same person > 0.3 or 0.4 usually. 
        # distance = 1 - sim?
        # Let's assume threshold is similarity.
        if best_sim > self.tolerance:
             return self.known_names[best_idx], float(best_sim)
             
        return "Unknown", float(best_sim)
