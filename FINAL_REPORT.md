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

Team VacTon developed a real-time face recognition security system running on Raspberry Pi 5. The system detects faces using MediaPipe, identifies authorized users with ArcFace recognition, and sends Telegram alerts when strangers are detected. We implemented a Face Mesh visualization overlay and achieved 18-20 FPS performance through systematic optimization including detector replacement, threaded recognition, and ROI-based processing.

---

## II. COMPLETED TASKS (What we did)

**1. Face Detection System**
*   **Member(s):** All
*   **Description:** Implemented MediaPipe Face Detection to locate faces in video feed. Replaced initial YOLO implementation with MediaPipe for better CPU performance.
*   **Outcome:** Detection runs at 8-10ms per frame, enabling real-time processing.

**2. Face Recognition with ArcFace**
*   **Member(s):** All
*   **Description:** Integrated pre-trained ArcFace model (w600k_r50.onnx) for face identification. Implemented embedding comparison with configurable confidence threshold (0.40).
*   **Outcome:** Accurate recognition of registered users with 512-dimensional face embeddings.

**3. Face Mesh Visualization**
*   **Member(s):** All
*   **Description:** Added MediaPipe Face Mesh to draw wireframe overlay on detected faces, including contour lines and nose crosshair marker.
*   **Outcome:** Visual feedback showing system is actively tracking faces.

**4. Performance Optimization**
*   **Member(s):** All
*   **Description:** Implemented three optimization strategies:
    *   Replaced YOLO detector (550ms) with MediaPipe (10ms)
    *   Added ROI cropping for Face Mesh processing
    *   Moved recognition to background thread (AsyncRecognizer class)
*   **Outcome:** Improved from 3-4 FPS to 18-20 FPS with all features enabled.

**5. Identity Deduplication**
*   **Member(s):** All
*   **Description:** Added logic to prevent multiple faces from being assigned the same identity. When conflicts occur, highest confidence match keeps the identity, others marked as "Unknown".
*   **Outcome:** Eliminated false positives in multi-person scenarios.

**6. Telegram Notifications**
*   **Member(s):** All
*   **Description:** Integrated Telegram bot to send alerts with photos when strangers are detected. Includes 10-second cooldown to prevent spam.
*   **Outcome:** Real-time alerts to home owner's phone.

**7. Data Collection Tools**
*   **Member(s):** All
*   **Description:** Created `tools/add_user.py` for automated photo capture (150 images in burst mode) and `tools/process_database.py` for embedding generation.
*   **Outcome:** Simplified user registration process.

**8. WiFi Configuration**
*   **Member(s):** All
*   **Description:** Configured NetworkManager to support multiple WiFi networks (home + mobile hotspot) for classroom demo.
*   **Outcome:** System can connect to available network automatically.

---

## III. IN-PROGRESS / ONGOING TASKS

None - all planned features are implemented and functional.

---

## IV. OUTCOMES & ACHIEVEMENTS

**1. Real-Time Performance**
Achieved 18-20 FPS with three concurrent AI models (detection, recognition, mesh visualization) running on Raspberry Pi 5 CPU.

**2. Identity Management**
Implemented deduplication system ensuring each identity appears only once per frame, preventing confusion in multi-person scenarios.

**3. User-Friendly Tools**
Created automated data collection (`add_user.py`) reducing registration from manual photo taking to 10-second capture session.

**4. Production Deployment**
Configured multi-network WiFi support enabling demo in classroom environment without reconfiguration.

**5. Performance Metrics**
- Detection: 8-10ms (MediaPipe)
- Recognition: 150ms (background thread)
- Mesh rendering: 30ms per face
- Overall FPS: 18-20 (single person), 11-15 (3 people)

---

## V. CHALLENGES, RISKS & MITIGATION

**Challenge 1: Low Frame Rate**
*   **Problem:** Initial implementation achieved only 3-4 FPS.
*   **Cause:** YOLO detector consumed 550ms per frame.
*   **Solution:** Replaced with MediaPipe detector (10ms), achieving 50x speedup.

**Challenge 2: Identity Flickering**
*   **Problem:** Face identities would change randomly between frames.
*   **Cause:** Tracking system lost identity when confidence was reset during deduplication.
*   **Solution:** Preserved confidence scores while changing name to "Unknown".

**Challenge 3: Duplicate Identities**
*   **Problem:** Two people could be labeled with same name.
*   **Cause:** Similar-looking people both matched same identity above threshold.
*   **Solution:** Implemented deduplication logic keeping highest confidence match.

---

## VI. PLAN FOR NEXT WEEK

| Task | Responsible | Deliverable |
| :--- | :--- | :--- |
| **Live Demo Execution** | All | Successful classroom presentation |
| **Performance Documentation** | All | FPS measurements across scenarios |
| **Code Cleanup** | All | Remove unused files, add comments |

---

## VII. REFERENCES

1. Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*. CVPR.
2. Lugaresi, C., et al. (2019). *MediaPipe: A Framework for Building Perception Pipelines*. arXiv:1906.08172.
