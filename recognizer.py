import face_recognition
import os
import logging
import pickle

class Recognizer:
    def __init__(self, config):
        self.faces_dir = config["paths"]["faces_dir"]
        self.tolerance = config["system"]["recognition_tolerance"]
        self.known_face_encodings = []
        self.known_face_names = []

    def load_known_faces(self):
        """Loads face images from subdirectories in faces_dir."""
        logging.info("Loading known faces...")
        
        if not os.path.exists(self.faces_dir):
            logging.warning(f"Faces directory '{self.faces_dir}' not found. Creating it.")
            os.makedirs(self.faces_dir, exist_ok=True)
            return

        for person_name in os.listdir(self.faces_dir):
            person_dir = os.path.join(self.faces_dir, person_name)
            
            if not os.path.isdir(person_dir):
                continue

            for filename in os.listdir(person_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filepath = os.path.join(person_dir, filename)
                    try:
                        image = face_recognition.load_image_file(filepath)
                        encodings = face_recognition.face_encodings(image)
                        
                        if len(encodings) > 0:
                            # We assume the first face found is the correct one
                            self.known_face_encodings.append(encodings[0])
                            self.known_face_names.append(person_name)
                            logging.debug(f"Loaded face for {person_name} from {filename}")
                        else:
                            logging.warning(f"No face found in {filename}")
                    except Exception as e:
                        logging.error(f"Error processing {filename}: {e}")
        
        logging.info(f"Loaded {len(self.known_face_names)} known faces.")

    def identify(self, face_image):
        """
        Identifies a face image (numpy array).
        Returns the name of the person or 'Unknown'.
        """
        # Encode the unknown face
        # Note: face_recognition expects RGB. OpenCV is BGR.
        # Ensure the input image is already RGB or convert it here if needed.
        # We will assume the main loop passes RGB for efficiency or we convert here.
        # face_locations is passed as None to let the library find it, 
        # BUT since we already have the bounding box from YOLO, 
        # we could optimize this by passing the location.
        # For now, let's re-encode fully to be safe or just encode.
        
        try:
            unknown_encodings = face_recognition.face_encodings(face_image)
            
            if not unknown_encodings:
                return "Unknown"

            # Use the first face found in the crop
            unknown_encoding = unknown_encodings[0]

            matches = face_recognition.compare_faces(self.known_face_encodings, unknown_encoding, tolerance=self.tolerance)
            name = "Unknown"

            # Or use face_distance to find the best match
            face_distances = face_recognition.face_distance(self.known_face_encodings, unknown_encoding)
            
            if len(face_distances) > 0:
                import numpy as np
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            return name
            
        except Exception as e:
            logging.error(f"Recognition error: {e}")
            return "Error"
