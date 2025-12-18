
import os
import sys
import subprocess
import time

# --- Worker Script Content ---
WORKER_SCRIPT = r"""
import os
import sys
import requests
import shutil
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def download_file(url, filename):
    print(f"Downloading {filename} from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(8192):
                f.write(chunk)
        print(f"Downloaded {filename}")
        return True
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        return False

def main():
    print("=== Worker Started: Exporting Models ===")
    
    import torch
    import numpy as np
    import onnx
    import insightface
    from ultralytics import YOLO
    
    print(f"Numpy version: {np.__version__}")
    print(f"Insightface version: {insightface.__version__}")

    # --- 1. Export Detector (YOLOv8-Face) ---
    print("\n[1/2] Processing Detector (YOLOv8n-Face)...")
    face_model_url = "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt"
    face_model_pt = "yolov8n-face.pt"
    
    if not os.path.exists(face_model_pt):
        if not download_file(face_model_url, face_model_pt):
            print("Using fallback yolov8n.pt (standard) just to ensure workflow completion...")
            face_model_pt = "yolov8n.pt"

    print(f"Exporting {face_model_pt} to ONNX...")
    try:
        model = YOLO(face_model_pt)
        # Export with fixed size for Pi 5 NPU/CPU stability
        export_path = model.export(format="onnx", imgsz=640, dynamic=False)
        print(f"Detector SUCCESS: {export_path}")
    except Exception as e:
        print(f"Detector Export Failed: {e}")

    # --- 2. Export Recognizer (ArcFace) ---
    print("\n[2/2] Processing Recognizer (InsightFace - Buffalo_S/L)...")
    
    # We use buffalo_l (ResNet50) for accuracy, or buffalo_s (MobileNet) for speed.
    # Pi 5 can handle ResNet50 typically.
    model_pack = 'buffalo_l' 
    
    try:
        print(f"Initializing FaceAnalysis with {model_pack}...")
        # providers=['CPUExecutionProvider'] essential for Kaggle CPU envs to avoid CUDA errors if not present
        app = insightface.app.FaceAnalysis(name=model_pack, providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Locate the downloaded file
        # InsightFace downloads to ~/.insightface/models/buffalo_l/w600k_r50.onnx
        home_dir = os.path.expanduser("~")
        model_root = os.path.join(home_dir, ".insightface", "models", model_pack)
        
        # Possible names depending on pack
        possible_names = ["w600k_r50.onnx", "2d106det.onnx", "w600k_mbf.onnx"]
        found = False
        
        if os.path.exists(model_root):
            for fname in os.listdir(model_root):
                if fname.endswith(".onnx") and "det" not in fname and "gender" not in fname:
                    # Usually the largest one is the recognition model
                    # For buffalo_l: w600k_r50.onnx
                    # For buffalo_s: w600k_mbf.onnx
                    src = os.path.join(model_root, fname)
                    dst = fname
                    shutil.copy(src, dst)
                    print(f"Recognizer SUCCESS: Copied {src} to {dst}")
                    found = True
        
        if not found:
            print(f"Could not automatically identify the recognition ONNX in {model_root}")
            print("Listing directory:")
            for root, dirs, files in os.walk(model_root):
                 for f in files: print(os.path.join(root, f))
                 
    except Exception as e:
        print(f"Recognizer Setup Failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== Export Complete ===")
    print("Files to download:")
    print("1. *.onnx (Detector)")
    print("2. *.onnx (Recognizer)")

if __name__ == "__main__":
    main()
"""

def install_dependencies():
    print("Installing dependencies (this may take a minute)...")
    # Pin numpy<2.0 to avoid SciPy/InsightFace conflicts on Colab/Kaggle
    pkgs = [
        "numpy<2.0", 
        "scipy", 
        "ultralytics", 
        "insightface", 
        "onnx", 
        "onnxruntime"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + pkgs)

def run_worker():
    # Write the worker script to a file
    worker_filename = "export_worker_temp.py"
    with open(worker_filename, "w") as f:
        f.write(WORKER_SCRIPT)
    
    print("Running worker script in a fresh process...")
    try:
        subprocess.check_call([sys.executable, worker_filename])
    except subprocess.CalledProcessError as e:
        print(f"Error running worker: {e}")
    finally:
        if os.path.exists(worker_filename):
            os.remove(worker_filename)

if __name__ == "__main__":
    install_dependencies()
    print("\nDependencies installed. Launching logic...")
    run_worker()
