# FINAL PROJECT REPORT

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

Team VacTon successfully developed a production-ready AI-powered home security system that achieves **20 FPS real-time performance** on Raspberry Pi 5 hardware. The system features a futuristic **Face Mesh visualization** ("Iron Man HUD"), **multi-person face recognition** with identity deduplication, and **instant Telegram alerts** for unauthorized access. 

Through systematic optimization, we replaced the initial YOLO detector with MediaPipe (achieving 50x speed improvement), implemented threaded architecture for concurrent processing, and added intelligent identity conflict resolution. The final system demonstrates that edge AI can deliver both visual sophistication and computational efficiency when properly architected.

---

## II. COMPLETED TASKS (What we did)

**1. Face Mesh Integration (Visualization Layer)**
*   **Member(s):** All
*   **Description:** Integrated Google MediaPipe Face Mesh (468 landmarks) to overlay a cyan wireframe on detected faces, creating a "cyberpunk" aesthetic.
*   **Outcome:** Real-time 3D head tracking with optimized ROI processing to maintain performance.

**2. Performance Optimization (Multi-Stage)**
*   **Member(s):** All
*   **Description:** Systematic optimization through three major iterations:
    *   **Stage 1:** ROI Cropping - Process only detected face regions instead of full frame
    *   **Stage 2:** Threading Architecture - Moved ArcFace recognition to background thread
    *   **Stage 3:** Detector Replacement - Switched from YOLO (550ms) to MediaPipe (10ms)
*   **Outcome:** Achieved **18-20 FPS** with all features enabled (detection + recognition + mesh).

**3. Identity Deduplication System**
*   **Member(s):** All
*   **Description:** Implemented conflict resolution logic to prevent multiple faces from being assigned the same identity. When conflicts occur, the highest confidence match retains the identity while others are marked as "Unknown".
*   **Outcome:** Eliminated false positives in multi-person scenarios, ensuring each home owner identity is unique per frame.

**4. Data Collection & Processing Tools**
*   **Member(s):** All
*   **Description:** Created `tools/add_user.py` for automated burst capture (150 photos) and `tools/clean_dataset.py` for geometry-based quality filtering.
*   **Outcome:** Streamlined user onboarding process from manual photo collection to automated 10-second capture session.

**5. Custom Model Training R&D**
*   **Member(s):** All
*   **Description:** Built complete training pipeline on Kaggle using PyTorch, trained both ArcFace and Triplet Loss architectures from scratch.
*   **Outcome:** Validated that pre-trained models (5.8M images) outperform custom models (100 images) for security applications. Educational success demonstrating data-centric AI principles.

**6. System Deployment Configuration**
*   **Member(s):** All
*   **Description:** Configured multi-network WiFi support for seamless demo deployment, enabling automatic connection to mobile hotspot or home network.
*   **Outcome:** Production-ready system that can operate in classroom environment without reconfiguration.

---

## III. IN-PROGRESS / ONGOING TASKS

**1. Hailo-8L NPU Integration**
*   **Member(s):** [Name 1]
*   **Status:** Research Phase (60% Complete)
*   **Description:** Investigating hardware acceleration for detection model to push FPS beyond 30.
*   **Note:** Current CPU-optimized MediaPipe solution already meets performance requirements.

**2. Final Presentation Rehearsal**
*   **Member(s):** All
*   **Status:** 90% Complete
*   **Description:** Practiced live demo transitions, prepared backup scenarios, tested mobile hotspot connectivity.

---

## IV. OUTCOMES & ACHIEVEMENTS

**1. Real-Time Multi-Model AI Pipeline**
Successfully orchestrated three concurrent AI models (MediaPipe Detection, ArcFace Recognition, Face Mesh Visualization) on edge hardware, achieving 20 FPS through architectural optimization rather than hardware acceleration.

**2. Production-Grade Identity Management**
Implemented enterprise-level features including:
- Identity deduplication (no duplicate names)
- Confidence-based conflict resolution
- Temporal tracking for smooth transitions
- Stranger detection with cooldown logic

**3. Developer-Friendly Tooling**
Created reusable utilities:
- `add_user.py`: Automated data collection (150 photos in 10 seconds)
- `clean_dataset.py`: Geometry-based quality filtering using Face Mesh
- `process_database.py`: One-command embedding generation

**4. Educational Deep Learning Insights**
Through custom model training experiments, we empirically validated the "data-centric AI" principle: **model architecture matters less than training data quality and quantity**. Our custom models (100 images) achieved 65% accuracy while pre-trained models (5.8M images) achieved 95% accuracy.

**5. Performance Metrics**
- **FPS:** 18-20 (3 people), 25+ (1 person)
- **Detection Latency:** 8-10ms (MediaPipe)
- **Recognition Latency:** 150ms (background thread, non-blocking)
- **Mesh Rendering:** 30ms per face

---

## V. CHALLENGES, RISKS & MITIGATION

**Challenge 1: Frame Rate Bottleneck**
*   **Risk:** Initial implementation achieved only 3-4 FPS, making the system unusable.
*   **Root Cause:** YOLO detection consumed 550ms per frame.
*   **Mitigation:** Replaced YOLO with MediaPipe (10ms), achieving 50x speedup. Lesson: Choose models optimized for target hardware.

**Challenge 2: Identity Flickering**
*   **Risk:** Faces would randomly switch between different identities frame-to-frame.
*   **Root Cause:** Tracking system lost identity when confidence was reset to 0.0 during deduplication.
*   **Mitigation:** Preserved original confidence scores while changing name to "Unknown", maintaining tracking stability.

**Challenge 3: False Positives (Duplicate Names)**
*   **Risk:** Similar-looking people would both be recognized as the same person.
*   **Root Cause:** Low recognition threshold (0.40) allowed weak matches.
*   **Mitigation:** Implemented deduplication system that resolves conflicts by confidence ranking.

**Challenge 4: Custom Model Underperformance**
*   **Risk:** Invested significant time in custom training but achieved poor results.
*   **Root Cause:** Insufficient training data (100 images vs. 5.8M for pre-trained).
*   **Mitigation:** Pivoted to pre-trained model. Lesson: Data quantity trumps model complexity for face recognition.

---

## VI. PLAN FOR NEXT WEEK

| Task | Responsible | Deliverable |
| :--- | :--- | :--- |
| **Live Demo Execution** | All | Successful classroom presentation with mobile hotspot |
| **Performance Documentation** | [Name 1] | FPS benchmarks across different scenarios |
| **Code Documentation** | [Name 2] | Inline comments and README updates |
| **Future Roadmap** | [Name 3] | Hailo NPU integration plan, cloud storage options |

---

## VII. REFERENCES

1. Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*. CVPR.
2. GoogleAI. (2020). *MediaPipe Face Mesh*. GitHub. https://github.com/google/mediapipe
3. Lugaresi, C., et al. (2019). *MediaPipe: A Framework for Building Perception Pipelines*. arXiv:1906.08172.
4. Schroff, F., Kalenichenko, D., & Philbin, J. (2015). *FaceNet: A Unified Embedding for Face Recognition and Clustering*. CVPR.
