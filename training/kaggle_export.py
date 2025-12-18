
import os
import sys
import subprocess
import time
import shutil

# --- Worker Script Content ---
WORKER_SCRIPT = r"""
import os
import sys
import requests
import shutil
import warnings

# Suppress all warnings to keep output clean
warnings.filterwarnings("ignore")

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
    
    # Imports inside the protected environment
    import torch
    import numpy as np
    import onnx
    import insightface
    from ultralytics import YOLO
    
    print(f"Environment Check: Numpy {np.__version__}, InsightFace {insightface.__version__}")

    # --- 1. Export Detector (YOLOv8-Face) ---
    print("\n[Step 1/2] Processing Detector (YOLOv8n-Face)")
    face_model_url = "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt"
    face_model_pt = "yolov8n-face.pt"
    
    if not os.path.exists(face_model_pt):
        if not download_file(face_model_url, face_model_pt):
             # Fallback if specific face model download fails
            print(" ! Warning: Download failed. Using generic yolov8n.pt as placeholder.")
            face_model_pt = "yolov8n.pt"

    print(f"Exporting {face_model_pt} to ONNX...")
    try:
        model = YOLO(face_model_pt)
        # Exporting with fixed size (640) for best stability
        path = model.export(format="onnx", imgsz=640, dynamic=False)
        print(f" -> Detector Success: {path}")
    except Exception as e:
        print(f" -> Detector Failed: {e}")

    # --- 2. Export Recognizer (ArcFace) ---
    print("\n[Step 2/2] Processing Recognizer (Buffalo_L)")
    
    # Using 'buffalo_l' (ResNet50) for best accuracy on Pi 5
    model_pack = 'buffalo_l' 
    
    try:
        # Force CPU provider
        app = insightface.app.FaceAnalysis(name=model_pack, providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Locate the underlying ONNX file
        home = os.path.expanduser("~")
        model_root = os.path.join(home, ".insightface", "models", model_pack)
        found = False
        
        if os.path.exists(model_root):
            for fname in os.listdir(model_root):
                # w600k_r50.onnx is the recognition model in buffalo_l
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
    print("Check the 'Output' file browser on the right for:")
    print(" 1. yolov8n-face.onnx (or similar)")
    print(" 2. w600k_r50.onnx (or similar)")

if __name__ == "__main__":
    main()
"""


def setup_and_run():
    print("=== Setting up Isolated Environment (Virtual Environment) ===")
    
    venv_dir = os.path.abspath("temp_venv")
    
    # 1. Create Venv
    # We use --system-site-packages=False (default) to isolate
    if os.path.exists(venv_dir):
        shutil.rmtree(venv_dir)
        
    print(f"Creating venv in {venv_dir}...")
    try:
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
    except Exception as e:
        print(f"Failed to create venv: {e}")
        return

    # 2. Determine Python Executable in Venv
    if os.name == "nt":
        venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        # On Linux/Unix, it could be python or python3
        p1 = os.path.join(venv_dir, "bin", "python")
        p2 = os.path.join(venv_dir, "bin", "python3")
        if os.path.exists(p1):
            venv_python = p1
        elif os.path.exists(p2):
            venv_python = p2
        else:
            print("Error: Could not find python binary in venv/bin")
            print(f"Listing {os.path.join(venv_dir, 'bin')}:")
            try:
                print(os.listdir(os.path.join(venv_dir, 'bin')))
            except:
                print("Could not list bin dir.")
            return

    print(f"Using venv python: {venv_python}")

    # 3. Install dependencies via 'python -m pip'
    # This is safer than calling 'pip' directly which might be missing
    print("Installing libraries...")
    pkgs = ["numpy<2.0", "onnx", "onnxruntime", "ultralytics", "insightface", "scipy"]
    
    try:
        subprocess.check_call([venv_python, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([venv_python, "-m", "pip", "install"] + pkgs)
    except Exception as e:
        print(f"Failed to install dependencies: {e}")
        return
    
    # 4. Write and Run Worker
    worker_path = os.path.abspath("worker.py")
    with open(worker_path, "w") as f:
        f.write(WORKER_SCRIPT)
        
    print("\n=== Launching Worker Script ===")
    sys.stdout.flush()
    subprocess.check_call([venv_python, worker_path])

if __name__ == "__main__":
    setup_and_run()
