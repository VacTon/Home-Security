# FINAL PRESENTATION SCRIPT (4-Person Team)
**Project:** AI-Powered Home Security System (Raspberry Pi 5)  
**Duration:** 5-7 Minutes  
**Demo Environment:** Classroom with Mobile Hotspot

---

## SPEAKER ASSIGNMENTS
- **Speaker 1 (Bao):** Introduction & System Overview
- **Speaker 2 (Phat):** Performance Optimization
- **Speaker 3 (Tien):** Face Recognition & Identity Management
- **Speaker 4 ([Name 4]):** Live Demo & Conclusion

---

## 1. INTRODUCTION & SYSTEM OVERVIEW (Speaker 1 - Bao)
**Time:** 0:00 - 1:30

**(Show live camera feed or screenshot)**

**Speaker 1:**
"Good morning, Professor and classmates. We are Team VacTon presenting our AI-Powered Home Security System.

Our system runs on a Raspberry Pi 5 and performs three main functions:
1. **Detects faces** in the camera feed
2. **Recognizes** if they are authorized family members
3. **Sends alerts** via Telegram when strangers appear

**(Point to screen showing face with wireframe overlay)**

You can see the system is currently running. The cyan wireframe you see is a Face Mesh overlay that tracks facial features in real-time. The green box shows the person is recognized, with their name and confidence score displayed.

The challenge was making this run in real-time on low-power hardware. Speaker 2 will explain how we optimized for performance."

---

## 2. PERFORMANCE OPTIMIZATION (Speaker 2 - Phat)
**Time:** 1:30 - 3:00

**(Slide: Performance comparison - 3 FPS vs 20 FPS)**

**Speaker 2:**
"When we first built the system, it ran at only 3-4 frames per second—too slow for real-time security. We made three key optimizations:

**First: Detector Replacement**
We initially used YOLO for face detection, which took 550 milliseconds per frame. We replaced it with Google MediaPipe, which takes only 10 milliseconds—a 50x speedup.

**Second: ROI Processing**
Instead of processing the entire 640x480 frame for the Face Mesh, we crop to just the detected face region. This reduced processing time by 80%.

**Third: Threading**
We moved the face recognition to a background thread. The main thread handles visualization while recognition runs in parallel, so the display never freezes.

**(Show FPS counter on screen)**

The result: we now run at 18-20 frames per second with all features enabled—detection, recognition, and mesh visualization.

Speaker 3 will explain how the face recognition works."

---

## 3. FACE RECOGNITION & IDENTITY MANAGEMENT (Speaker 3 - Tien)
**Time:** 3:00 - 4:30

**(Slide: Face recognition pipeline diagram)**

**Speaker 3:**
"For face recognition, we use ArcFace—a pre-trained model that converts each face into a 512-dimensional embedding. We compare this against our database of known faces.

**(Show add_user.py demo or screenshot)**

To register a new user, we created a tool called `add_user.py`. It captures 150 photos in 10 seconds while you rotate your head. This gives the system multiple angles to learn from.

**(Slide: Identity deduplication example)**

One important feature we added is identity deduplication. If two people in the frame both match the same identity, the system keeps the highest confidence match and marks the other as 'Unknown'. This prevents false positives.

For example, if two similar-looking people are detected and both match 'John', the system will keep the one with 85% confidence as 'John' and mark the 55% match as 'Unknown'.

Now Speaker 4 will show you the live demo."

---

## 4. LIVE DEMO & CONCLUSION (Speaker 4 - [Name 4])
**Time:** 4:30 - 6:30

**(Switch to live camera feed)**

**Speaker 4:**
"Let's see the system in action.

**(Point to screen)**

Right now, I'm being detected and recognized. You can see:
- My name and confidence score (e.g., 'Bao 0.78')
- The Face Mesh wireframe tracking my face
- The FPS counter showing 19 frames per second

**(Another team member enters frame)**

Now [Name] is entering. The system detects both of us simultaneously and recognizes each person correctly.

**(Cover face or have stranger enter)**

If I cover my face or an unknown person appears...

**(System shows "Unknown" in red)**

The box turns red and shows 'Unknown'. Within 10 seconds, the system saves a photo and sends a Telegram alert.

**(Show Telegram notification on phone)**

Here's the alert on my phone with the photo of the unknown person.

**(Return to slides)**

**In summary:**
- We built a real-time face recognition system running at 20 FPS on Raspberry Pi 5
- We optimized performance by replacing the detector, using ROI processing, and threading
- We implemented identity deduplication to prevent false positives
- The system sends instant Telegram alerts for unknown persons

Thank you. We're ready for questions."

---

## Q&A PREPARATION

**Q: How accurate is the face recognition?**
*A:* "With good lighting and frontal faces, we achieve over 90% accuracy. The system requires at least 55% confidence to identify someone, which prevents false positives."

**Q: What happens if WiFi goes down?**
*A:* "The detection and recognition continue to work locally. Only the Telegram alerts require internet. We could add local storage as a backup."

**Q: Can it handle multiple people?**
*A:* "Yes, we've tested with 3 people simultaneously. FPS drops to about 11-12 with 3 faces due to the mesh rendering, but recognition remains accurate."

**Q: Why use Raspberry Pi instead of cloud processing?**
*A:* "Privacy and latency. All processing happens on-device, so no video is uploaded to external servers. Response time is also faster."

**Q: How long does it take to register a new user?**
*A:* "About 10 seconds using our `add_user.py` tool, which captures 150 photos automatically. Then you run the processing script which takes about 30 seconds."

---

## DEMO BACKUP PLAN

**If WiFi fails:**
- Show pre-recorded video of system working
- Display screenshots of Telegram alerts from previous tests

**If camera fails:**
- Use laptop webcam (system supports standard USB cameras)
- Show screenshots of working system

**If recognition fails:**
- Re-run `python tools/process_database.py` before presentation
- Have backup photos ready

---

## TIMING BREAKDOWN
- Introduction: 1.5 min
- Optimization: 1.5 min
- Recognition: 1.5 min
- Demo: 2 min
- **Total:** 6.5 minutes (leaves 3.5 min for Q&A)
