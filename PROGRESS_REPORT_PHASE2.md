# WEEKLY PROJECT REPORT

**Project Name:** AI-Powered Home Security System (Raspberry Pi 5)
**Supervisor:** [Supervisor Name]
**Module:** CSE2022 - Capstone Project
**Group Number:** VacTon
**Names:**
1. Bao
2. Phat
3. Tien
4. [Name 4]

---

## Table of Contents
I. ABSTRACT / PROJECT OVERVIEW  
II. COMPLETED TASKS (What we did)  
III. IN-PROGRESS / ONGOING TASKS  
IV. OUTCOMES & ACHIEVEMENTS  
V. CHALLENGES, RISKS & MITIGATION  
VI. PLAN FOR NEXT WEEK  
VII. REFERENCES  

---

## I. ABSTRACT / PROJECT OVERVIEW
This week, Team VacTon implemented a major User Interface upgrade by integrating real-time **Face Mesh visualization** ("Iron Man" HUD) into the security feed. Simultaneously, we conducted an intensive R&D experiment to train a **custom Face Recognition model** from scratch using Kaggle GPUs. While the visualization was a complete success, adding a premium futuristic aesthetic, the custom training experiment revealed that pre-trained large-scale models offer superior accuracy for security applications. Consequently, we finalized the system architecture to combine the new visual layer with the robust pre-trained recognition engine.

---

## II. COMPLETED TASKS (What we did)

**1. Face Mesh Integration (Visualization Layer)**
*   **Member(s):** [Name 1], [Name 2]
*   **Description:** Integrated Google MediaPipe Face Mesh (468 landmarks) to overlay a cyan wireframe on detected faces.
*   **Outcome:** Successfully running at 18-24 FPS on Raspberry Pi 5 using a custom Region-of-Interest (ROI) optimization strategy. The visual effect provides immediate feedback that the system is "active" and tracking.

**2. Custom Model Training Pipeline (R&D)**
*   **Member(s):** [Name 3], [Name 4]
*   **Description:** Built an end-to-end training pipeline on Kaggle using PyTorch. This involved creating `tools/clean_dataset.py` for automated data cleaning (geometry-based filtering) and training two architectures: ArcFace (ResNet18) and Triplet Loss (MobileNetV2).
*   **Outcome:** Successfully trained and exported `.onnx` models. Empirical testing confirmed that while functional, they lacked the robustness of the 5.8M-image pre-trained model.

**3. System Stabilization & "Blue Face" Fix**
*   **Member(s):** All
*   **Description:** Debugged a critical issue where the camera feed appeared blue, causing recognition failures. Identified an RGB/BGR mismatch between `Picamera2` and `OpenCV`.
*   **Outcome:** Fixed via manual channel swapping in `camera.py`. Recognition is now accurate.

---

## III. IN-PROGRESS / ONGOING TASKS

**1. Hardware Acceleration (Hailo-8L)**
*   **Member(s):** [Name 1]
*   **Status:** 60% Done (Research Phase)
*   **Description:** We are currently investigating moving the heavy ArcFace inference from the CPU to the Hailo-8L NPU. This is complex due to the need for model quantization (Float32 -> Int8) but is necessary to push the system from ~20 FPS to 60+ FPS.

**2. Final Presentation Rehearsal**
*   **Member(s):** All
*   **Status:** 20% Done
*   **Description:** We have drafted the initial script and assigned roles. We need to practice the live demo transition, specifically the moment the "Unknown" tag switches to a "Known Name" with the Face Mesh overlay.

---

## IV. OUTCOMES & ACHIEVEMENTS

**1. The "Iron Man HUD"**
We achieved a visually stunning, low-latency face mesh overlay. By cropping the processing area to the face bounding box, we avoided the typical lag associated with mesh generation on edge devices.

**2. The Data Cleaning Tool**
We released `tools/clean_dataset.py`, a utility that uses the Face Mesh to geometrically analyze training photos. It automatically rejects blurry or off-angle (yaw > 25°) photos. This is a reusable asset for future machine learning projects.

**3. The "Deep Learning Reality Check"**
We proved definitively that for our security use case, a **Pre-trained SOTA Model** (trained on MS1MV3) is superior to a custom-trained model. This "failure" was a significant educational achievement, saving us from deploying a sub-par custom model.

---

## V. CHALLENGES, RISKS & MITIGATION

**Challenge 1: Frame Rate Drops**
*   **Risk:** Adding Face Mesh dropped FPS to <5 initially.
*   **Mitigation:** Implemented **ROI Cropping**. We only calculate the mesh inside the detected face box. This regained ~15 FPS.

**Challenge 2: Custom Model Accuracy**
*   **Risk:** The custom model had a high False Rejection Rate.
*   **Mitigation:** We pivoted back to the **Pre-trained ArcFace Model**. We use the custom training pipeline only for educational demonstration, not production security.

---

## VI. PLAN FOR NEXT WEEK

| Task | Responsible | Deliverable |
| :--- | :--- | :--- |
| **Final System Stress Test** | [Name 1] | 24-hour stability run log. |
| **Hailo-8L Integration** | [Name 2] | Quantized .hef model (optional goal). |
| **Presentation Deck** | [Name 3] | Final PowerPoint slides. |
| **Live Demo Setup** | [Name 4] | Physical camera/lighting setup for demo. |

---

## VII. REFERENCES
1.  Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*. CVPR.
2.  GoogleAI. (2020). *MediaPipe Face Mesh*. GitHub.
