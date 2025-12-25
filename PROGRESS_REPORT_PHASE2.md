# WEEKLY PROGRESS REPORT

**Project Title:** Advanced Home Security System using Raspberry Pi 5 & Edge AI  
**Team:** VacTon (Bao, Phat)  
**Date:** December 25, 2025  
**Report ID:** Phase 2 - Visualization & Model Research

---

## 1. Objectives for this Week
*   Enhance the User Interface (UI) with real-time biometric visualization ("Face Mesh").
*   Investigate the feasibility of training a custom Face Recognition model to replace the pre-trained ArcFace solution.
*   Implement an end-to-end data pipeline: Data Collection -> Cleaning -> Training -> Deployment.

---

## 2. Work Completed

### 2.1 UI/UX Enhancement: Face Mesh Integration
*   **Objective:** Create a futuristic "Iron Man HUD" feel to improve the aesthetic value of the product.
*   **Implementation:** 
    *   Integrated `MediaPipe Face Mesh` (468 landmarks) into the main video loop.
    *   developed `visualizer.py` to handle drawing logic separately from the detection core.
*   **Optimization Strategy:**
    *   Implemented a **Region of Interest (ROI)** cropping mechanism.
    *   Instead of processing the full 1080p frame, we crop only the bounding box of the face, run the mesh, and project it back.
    *   **Result:** Maintained system stability with acceptable FPS on the Raspberry Pi 5.

### 2.2 R&D: Custom Model Training (The "Deep Learning Experiment")
*   **Objective:** Train a lightweight model (MobileNetV2/MobileFaceNet) on our specific home users to potentially improve speed/accuracy.
*   **Workflow:**
    1.  **Data Collection:** Gathered ~100 photos of team members.
    2.  **Data Cleaning:** Developed `tools/clean_dataset.py` which uses geometric analysis (Yaw/Pitch) to auto-reject bad angles and blurry photos.
    3.  **Training:** Utilized Kaggle GPUs to train two architectures:
        *   *Triplet Loss* (FaceNet approach).
        *   *ArcFace Loss* (Angular Margin approach).
    4.  **Deployment:** Exported models to ONNX and integrated them into the `recognizer.py` module.

---

## 3. Results & Analysis (The "Educational Win")

### 3.1 Face Mesh Success
The visualization layer works perfectly. The **Cyan Wireframe Overlay** accurately tracks facial movements (blinking, talking) in real-time, significantly boosting the "Wow Factor" for the final presentation.

### 3.2 Deep Learning Reality Check
We conducted a comparative analysis between our **Custom Trained Model** and the **Pre-trained ArcFace Model**.

| Metric | Pre-trained Model (Google/InsightFace) | Custom Model (Ours) |
| :--- | :--- | :--- |
| **Training Data** | 5.8 Million Images (MS1MV3) | ~100 Images (Home Data) |
| **Robustness** | High (Works in dark, angles, glasses) | Low (Overfits to specific lighting) |
| **Conclusion** | **Superior** | **Inferior** |

**Key Lesson Learned:**
Modern Face Recognition relies heavily on **Generalization**. A model trained on millions of faces learns the *abstract concept* of facial structure (how shadows fall on a nose, how eyes differ). A model trained on 50 photos merely "memorizes" those specific pixels. 

**Strategic Decision:**
We have decided to **revert to the Pre-trained ArcFace Model** for the production deployment. This ensures maximum security and accuracy, while we keep the Face Mesh and Data Cleaning tools as valuable R&D artifacts.

---

## 4. Challenges Encountered & Solutions

| Challenge | Impact | Solution |
| :--- | :--- | :--- |
| **Low FPS with Mesh** | System dropped to <5 FPS. | Implemented ROI Cropping (Face-only processing) to boost speed. |
| **"Blue Face" Bug** | Recognition failed; video looked blue. | Identified RGB/BGR mismatch in `camera.py`. Added manual channel swapping. |
| **Custom Model Inaccuracy** | High False Rejection Rate. | Analyzed "Small Data" problem. Reverted to Pre-trained SOTA weights. |

---

## 5. Plan for Next Week
*   **Final System Polish:** Ensure the system runs stable for 24+ hours.
*   **Presentation Prep:** Rehearse the live demo using the Face Mesh visualization.
*   **Documentation:** Finalize the "User Manual" and Architecture diagrams.

---

**Sign-off:**  
*Status: On Track for Final Presentation.*
