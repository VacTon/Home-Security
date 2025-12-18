
import os
import sys
import requests
import shutil

def download_file(url, filename):
    print(f"Downloading {filename} from {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"Downloaded {filename}")
    else:
        print(f"Failed to download {filename}")

def main():
    print("=== Kaggle Model Export Script ===")
    print("This script prepares your Face Recognition models for deployment on Raspberry Pi 5.")
    
    # 1. Install Dependencies
    print("\n[1] Installing dependencies...")
    os.system("pip install ultralytics insightface onnx onnxruntime")
    
    import torch
    from ultralytics import YOLO
    import insightface
    
    # 2. Export Detector (YOLOv8-Face)
    print("\n[2] Preparing Face Detector (YOLOv8n-Face)...")
    # URL for a pre-trained YOLOv8-Face model (generic)
    # Using Akanametov's release as a reliable source for YOLOv8-Face
    face_model_url = "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt"
    face_model_pt = "yolov8n-face.pt"
    
    if not os.path.exists(face_model_pt):
        download_file(face_model_url, face_model_pt)
    
    if os.path.exists(face_model_pt):
        print("Exporting to ONNX...")
        model = YOLO(face_model_pt)
        # Exporting with fixed size 640x640 is usually safest for Pi 5 inference
        export_path = model.export(format="onnx", imgsz=640, dynamic=False)
        print(f"Detector exported to: {export_path}")
    else:
        print("Error: Could not find/download yolov8n-face.pt")

    # 3. Export/Get Recognizer (ArcFace / Buffalo_S)
    print("\n[3] Preparing Face Recognizer (InsightFace - Buffalo_S)...")
    # We will initialize the FaceAnalysis app which automatically downloads models to ~/.insightface
    app = insightface.app.FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # Locate the model file. Buffalo_S includes 'w600k_mbf.onnx' (MobileFaceNet) or similar.
    # Actually 'buffalo_l' uses r50 (ResNet50). 'buffalo_s' uses lighter models.
    # For Pi 5, 'buffalo_s' is faster, but 'buffalo_l' (r50) is more accurate.
    # Let's try to get w600k_r50.onnx (from buffalo_l) for better security if Pi 5 can handle it.
    # If too slow, switch to buffalo_s.
    
    # Let's check where they are stored.
    home_dir = os.path.expanduser("~")
    insightface_dir = os.path.join(home_dir, ".insightface", "models")
    
    # We need to find the recognition specific model.
    # Usually named 'w600k_r50.onnx' or similar inside the model folder.
    
    # Let's download buffalo_l specifically to get the R50 model
    print("Downloading buffalo_l model pack (includes ResNet50)...")
    app_l = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app_l.prepare(ctx_id=0, det_size=(640, 640))
    
    source_model_path = os.path.join(insightface_dir, "buffalo_l", "w600k_r50.onnx")
    dest_model_path = "w600k_r50.onnx"
    
    if os.path.exists(source_model_path):
        shutil.copy(source_model_path, dest_model_path)
        print(f"Recognizer copied to: {dest_model_path}")
    else:
        print(f"Could not find {source_model_path}. Listing contents of {insightface_dir}...")
        # Fallback listing
        for root, dirs, files in os.walk(insightface_dir):
            for file in files:
                print(os.path.join(root, file))

    print("\n=== FINISHED ===")
    print("Please download the following files from the Output section:")
    print(f"1. {face_model_pt.replace('.pt', '.onnx')}")
    print(f"2. {dest_model_path}")

if __name__ == "__main__":
    main()
