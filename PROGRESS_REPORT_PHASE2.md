# Project Progress Report: Home Security Phase 2
**Project:** Raspberry Pi 5 AI Security System  
**Date:** December 25, 2025  
**Topic:** Face Mesh Integration & Custom Model Research

---

## 1. Executive Summary
This week focused on elevating the User Interface (UI) and exploring the boundaries of Edge AI training. We successfully implemented a real-time **"Iron Man" Face Mesh** visualization, adding a futuristic HUD overlay.

Simultaneously, we undertook an ambitious Research & Development (R&D) initiative to train a **Custom Face Recognition Model** from scratch using Kaggle GPUs. While the training pipeline was successfully built, empirical testing revealed the **"Data Scarcity Principle,"** validating that pre-trained large-scale models outperform custom models trained on small datasets.

---

## 2. Key Achievements

### A. Real-Time Face Mesh (The "Iron Man" HUD)
We integrated Google's **MediaPipe Face Mesh** into the security feed.
*   **Optimization:** Implemented a "Region of Interest" (ROI) strategy to crop strictly around the detected face, allowing the mesh to run at **20+ FPS** on the Raspberry Pi 5.
*   **Visualization:** Designed a custom wireframe overlay (Cyan/Teal) to distinguish the secure system from standard bounding boxes.
*   **Outcome:** The system now looks premium and futuristic, meeting the "Wow Factor" requirement.

### B. Automated Dataset Curation (The "Smart Cleaner")
To support custom training, we developed a novel tool (`tools/clean_dataset.py`) that uses Face Mesh geometry to:
1.  **Calculate Yaw/Pitch:** Automatically reject profile shots (looking sideways).
2.  **Detect Blur:** Reject low-quality frames.
3.  **Auto-Align:** Rotationally correct faces based on eye calibration.
*   **Outcome:** We transformed a raw, noisy dataset into a "Golden Dataset" suitable for machine learning.

### C. End-to-End Training Pipeline
We successfully built and executed widespread industry training workflows on **Kaggle**:
*   **Architecture 1:** Triplet Loss (FaceNet style).
*   **Architecture 2:** ArcFace (State-of-the-Art discrimination).
*   **Deployment:** Successfully exported specialized `.onnx` models and deployed them to the edge device.

---

## 3. The "Deep Learning Reality Check" (Lessons Learned)

The primary goal of the R&D phase was to see if a custom-trained model could outperform the generic pre-trained model.

**The Experiment:**
*   **Model A (Pre-trained):** Trained on MS1MV3 (5.8 Million Images, 93k Identities).
*   **Model B (Custom):** Trained on Home Dataset (~100 Images, 3-4 Identities).

**The Result:**
Model A (Pre-trained) significantly outperformed Model B.

**The "Educational Win":**
We learned a fundamental truth of Modern AI: **Data is King.**
*   Deep Learning models require massive variability (lighting, ethnicity, age, texture) to learn the abstract concept of "What makes a face unique?"
*   Training on a small dataset leads to **Overfitting** (memorizing the specific training photos) rather than **Generalization** (recognizing the person in new conditions).

While the custom model "worked" (it learned!), it lacked the robust invariance of the Google-scale pre-trained model.

---

## 4. Strategic Decision
**Decision:** Revert to the **Pre-trained ArcFace Model (w600k_r50.onnx)**.

**Rationale:**
1.  **Security First:** Accuracy is the paramount metric for a security system.
2.  **Best of Both Worlds:** We keep the visual enhancements (Face Mesh) and the data cleaning tools, but we power the recognition engine with the SOTA weights.

---

## 5. Next Steps
1.  **Hardware Acceleration:** Move the Pre-trained model inference to the **Hailo-8L NPU** (if available) to boost FPS for the heavy ArcFace model.
2.  **Final Presentation:** Prepare the live demo script.
