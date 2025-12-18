
import os
import sys
import subprocess
import time
import shutil


# --- Worker Script Content ---
WORKER_SCRIPT = r"""
import os
import sys
import warnings

# CRITICAL: Add local libs to path BEFORE system libs
# This mimics a virtual environment
local_libs = os.path.abspath("temp_libs")
sys.path.insert(0, local_libs)

# Suppress warnings
warnings.filterwarnings("ignore")

import requests
import shutil

def download_file(url, filename):
    print(f"Downloading {filename}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(8192):
                f.write(chunk)
        print(f" -> Done.")
        return True
    except Exception as e:
        print(f" -> Failed: {e}")
        return False

def main():
    print("\n=== Worker Process: Starting Export ===")
    
    try:
        import numpy as np
        print(f"Using Numpy Version: {np.__version__}")
        print(f"Numpy Location: {os.path.dirname(np.__file__)}")
    except ImportError as e:
        print(f"CRITICAL ERROR: Could not import numpy from temp_libs: {e}")
        sys.exit(1)

    import torch
    import onnx
    import insightface
    from ultralytics import YOLO
    
    print(f"Environment Check: InsightFace {insightface.__version__}")

    # --- 1. Export Detector (YOLOv8-Face) ---
    print("\n[Step 1/2] Processing Detector (YOLOv8n-Face)")
    face_model_url = "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt"
    face_model_pt = "yolov8n-face.pt"
    
    if not os.path.exists(face_model_pt):
        if not download_file(face_model_url, face_model_pt):
            print(" ! Warning: Download failed. Using generic yolov8n.pt as placeholder.")
            face_model_pt = "yolov8n.pt"

    print(f"Exporting {face_model_pt} to ONNX...")
    try:
        model = YOLO(face_model_pt)
        path = model.export(format="onnx", imgsz=640, dynamic=False)
        print(f" -> Detector Success: {path}")
    except Exception as e:
        print(f" -> Detector Failed: {e}")

    # --- 2. Export Recognizer (ArcFace) ---
    print("\n[Step 2/2] Processing Recognizer (Buffalo_L)")
    model_pack = 'buffalo_l' 
    
    try:
        app = insightface.app.FaceAnalysis(name=model_pack, providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        home = os.path.expanduser("~")
        model_root = os.path.join(home, ".insightface", "models", model_pack)
        found = False
        
        if os.path.exists(model_root):
            for fname in os.listdir(model_root):
                if fname.endswith(".onnx") and "det" not in fname and "gender" not in fname:
                    shutil.copy(os.path.join(model_root, fname), fname)
                    print(f" -> Recognizer Success: Copied {fname}")
                    found = True
        
        if not found:
             print(" ! Error: Could not locate .onnx file in insightface cache.")
             
    except Exception as e:
        print(f" -> Recognizer Failed: {e}")

    print("\n=======================================")
    print("       EXPORT COMPLETED ")
    print("=======================================")
    print("Check the 'Output' file browser for your files!")

if __name__ == "__main__":
    main()
"""

def setup_and_run():
    print("=== Setting up Local Library Environment ===")
    print("Using 'pip install --target' to bypass system limitation...")
    
    libs_dir = os.path.abspath("temp_libs")
    
    # Clean cleanup
    if os.path.exists(libs_dir):
        try:
            shutil.rmtree(libs_dir)
        except:
            pass
    os.makedirs(libs_dir, exist_ok=True)
    
    # Packages
    # We install into libs_dir ignoring system packages to ensure we get the right versions
    pkgs = ["numpy<2.0", "onnx", "onnxruntime", "ultralytics", "insightface", "scipy"]
    
    print("Downloading and Installing libraries (Large Download)...")
    try:
        # --ignore-installed is key here to overwrite/ignore failing system constraints
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                               "--target", libs_dir, 
                               "--ignore-installed"] + pkgs)
    except Exception as e:
        print(f"Installation failed: {e}")
        return

    # Write worker
    with open("worker.py", "w") as f:
        f.write(WORKER_SCRIPT)
        
    print("\n=== Launching Worker Script ===")
    sys.stdout.flush()
    # Run using the SAME python interpreter, but the script puts 'temp_libs' first in path
    subprocess.check_call([sys.executable, "worker.py"])

if __name__ == "__main__":
    setup_and_run()
