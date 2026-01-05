import cv2
import mediapipe as mp
import numpy as np
import os
import shutil
import math

# ==========================================
# CONFIGURATION
# ==========================================
MAX_YAW_DEGREE = 25   # Reject if looking sideways > 25 degrees
MAX_PITCH_DEGREE = 25 # Reject if looking up/down > 25 degrees
MIN_SHARPNESS = 50    # Reject blurry images (Variance of Laplacian)
OUTPUT_SIZE = (112, 112) # Standard ArcFace input size

class DatasetCleaner:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True, # Improved accuracy for static images
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def get_sharpness(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def estimate_pose_and_align(self, image):
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            return None, "No Face Detected"

        landmarks = results.multi_face_landmarks[0].landmark
        h, w, c = image.shape

        # Key Landmarks
        nose_tip = landmarks[1]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        chin = landmarks[152]
        left_cheek = landmarks[234]
        right_cheek = landmarks[454]

        # --- 1. POSE CHECK (Simple Geometry) ---
        # Normalize coordinates
        nx, ny = nose_tip.x * w, nose_tip.y * h
        lx, ly = left_cheek.x * w, left_cheek.y * h
        rx, ry = right_cheek.x * w, right_cheek.y * h

        # Yaw: Ratio of Nose-to-Cheek distances
        left_dist = math.hypot(nx - lx, ny - ly)
        right_dist = math.hypot(nx - rx, ny - ry)
        
        # Avoid division by zero
        if right_dist == 0: right_dist = 0.001
        ratio = left_dist / right_dist
        
        # Crude Yaw approximation: 
        # Ratio 1.0 = Center. 
        # Ratio > 2.0 or < 0.5 implies side view.
        if ratio > 2.5 or ratio < 0.4:
            return None, f"Bad Yaw (Looking Sideways)"

        # --- 2. ALIGNMENT (Rotation) ---
        # Get vector between eyes
        ex_l, ey_l = left_eye.x * w, left_eye.y * h
        ex_r, ey_r = right_eye.x * w, right_eye.y * h
        
        # Calculate angle
        dY = ey_r - ey_l
        dX = ex_r - ex_l
        angle = np.degrees(np.arctan2(dY, dX)) - 180 # Correction might be needed depending on coord system
        # Actually standard atan2 for eyes: usually angle is small (~0).
        # Let's re-calc:
        angle = np.degrees(np.arctan2(ey_r - ey_l, ex_r - ex_l))
        
        # Rotate image to make eyes horizontal
        center = ((ex_l + ex_r) // 2, (ey_l + ey_r) // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned_img = cv2.warpAffine(image, M, (w, h))

        # --- 3. CROP ---
        # Re-run mesh on aligned image? No, strict crop to face box is faster.
        # We can simulate box from center.
        # Simple square crop around center point
        face_size = int(math.hypot(ex_r - ex_l, ey_r - ey_l) * 4.0) # Heuristic size
        cx, cy = int(center[0]), int(center[1])
        
        x1 = max(0, cx - face_size//2)
        y1 = max(0, cy - face_size//2)
        x2 = min(w, cx + face_size//2)
        y2 = min(h, cy + face_size//2)
        
        crop = aligned_img[y1:y2, x1:x2]
        
        try:
            crop = cv2.resize(crop, OUTPUT_SIZE)
            return crop, "Success"
        except:
            return None, "Crop failed"


def process_directory(input_dir="faces", output_dir="faces_clean"):
    cleaner = DatasetCleaner()
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    processed = 0
    rejected = 0

    # Walk through person folders
    for person_name in os.listdir(input_dir):
        person_path = os.path.join(input_dir, person_name)
        if not os.path.isdir(person_path): continue

        out_person_path = os.path.join(output_dir, person_name)
        os.makedirs(out_person_path, exist_ok=True)

        print(f"Processing {person_name}...")

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None: continue

                # 1. Blur Check
                sharpness = cleaner.get_sharpness(img)
                if sharpness < MIN_SHARPNESS:
                    print(f"  [REJECT] {img_name}: Blurry (Score {int(sharpness)})")
                    rejected += 1
                    continue

                # 2. Pose & Align
                cleaned_face, status = cleaner.estimate_pose_and_align(img)
                if cleaned_face is None:
                    print(f"  [REJECT] {img_name}: {status}")
                    rejected += 1
                    continue

                # Save Good Image
                cv2.imwrite(os.path.join(out_person_path, img_name), cleaned_face)
                processed += 1

            except Exception as e:
                print(f"Error {img_name}: {e}")

    print(f"\nDone! Processed: {processed}, Rejected: {rejected}")
    print(f"Clean dataset saved to: {output_dir}/")

if __name__ == "__main__":
    # You can change 'faces' to the path of your large raw dataset
    process_directory(input_dir="faces", output_dir="dataset_for_kaggle")
