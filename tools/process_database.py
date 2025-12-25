
import os
import sys
import cv2
import numpy as np
import logging

# Add parent dir to path to import project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detector import Detector
from recognizer import Recognizer
import yaml
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def process_dataset():
    """
    Scans 'raw_faces/' directory.
    Detects faces, aligns them, generates embeddings.
    Saves to 'faces/encodings.pkl'.
    """
    config = load_config()
    
    # Override paths for script execution context
    # config["paths"]["model_path"] = "../" + config["paths"]["model_path"]
    # We might need to handle relative paths carefully if running from 'tools/'
    # But usually models are absolute or relative to CWD.
    # Let's assume user runs this from the PROJECT ROOT: python tools/process_database.py
    
    # Helper to fix path if running from root
    if not os.path.exists("config.yaml"):
        print("Error: Please run this script from the project root directory!")
        print("Usage: python tools/process_database.py")
        return

    # Initialize Models
    print("Loading Models...")
    detector = Detector(config)
    recognizer = Recognizer(config)

    raw_dir = "faces" # User photos are here
    output_file = os.path.join(config["paths"]["faces_dir"], "encodings.pkl")
    
    if not os.path.exists(raw_dir):
        print(f"Creating {raw_dir}...")
        os.makedirs(raw_dir)
        print(f"Please put your photos in subfolders inside '{raw_dir}/' (e.g. {raw_dir}/Mom/photo1.jpg)")
        return

    known_encodings = []
    known_names = []
    
    print(f"Scanning '{raw_dir}'...")
    
    total_added = 0
    
    for person_name in os.listdir(raw_dir):
        person_dir = os.path.join(raw_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
            
        print(f" -> Processing {person_name}...")
        
        files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for filename in files:
            filepath = os.path.join(person_dir, filename)
            img = cv2.imread(filepath)
            if img is None:
                continue
            
            # 1. Detect Face
            # detector.detect returns list of dicts: {"box":..., "conf":..., "keypoints":...}
            detections = detector.detect(img)
            
            if not detections:
                print(f"    [Warning] No face found in {filename}. Skipping.")
                continue
            
            # Pick the largest face if multiple
            best_det = max(detections, key=lambda d: (d["box"][2]-d["box"][0]) * (d["box"][3]-d["box"][1]))
            
            # 2. Get Embedding (Aligned)
            # We pass the keypoints so recognizer aligns it perfectly
            emb = recognizer.get_embedding(img, kpts=best_det["keypoints"], box=best_det["box"])
            
            if emb is not None:
                # normalize is already done in get_embedding? 
                # recognizer.get_embedding returns normalized embedding (line 86)
                known_encodings.append(emb)
                known_names.append(person_name)
                total_added += 1
            else:
                print(f"    [Warning] Could not generate embedding for {filename}.")

    # Save
    if total_added > 0:
        print(f"Saving {total_added} faces to {output_file}...")
        data = {
            "encodings": known_encodings,
            "names": known_names
        }
        # Ensure dir exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "wb") as f:
            pickle.dump(data, f)
        print("Done! You can now run main.py.")
    else:
        print("No faces processed. Directory empty?")

if __name__ == "__main__":
    process_dataset()
