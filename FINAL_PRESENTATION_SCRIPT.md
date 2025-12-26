# FINAL PRESENTATION SCRIPT (4-Person Team)
**Project:** AI-Powered Home Security System (Raspberry Pi 5)  
**Duration:** 5-7 Minutes  
**Demo Environment:** Classroom with Mobile Hotspot

---

## SPEAKER ASSIGNMENTS
- **Speaker 1 (Bao):** Introduction & Problem Statement
- **Speaker 2 (Phat):** Technical Architecture & Optimization
- **Speaker 3 (Tien):** AI Models & Identity Management
- **Speaker 4 ([Name 4]):** Live Demo & Conclusion

---

## 1. INTRODUCTION & PROBLEM STATEMENT (Speaker 1 - Bao)
**Time:** 0:00 - 1:30

**(Slide: Title with "Iron Man HUD" Visual)**

**Speaker 1:**
"Good morning, Professor and classmates. We are Team VacTon, and today we're presenting our AI-Powered Home Security System.

Traditional security cameras are passive—they record footage but don't understand what they're seeing. Our goal was to build an **intelligent** system that:
1. **Recognizes** authorized family members in real-time
2. **Alerts** you instantly when strangers appear
3. **Looks futuristic** with a 'cyberpunk' interface

**(Slide: System Overview)**

But here's the challenge: We had to run three heavy AI models simultaneously on a Raspberry Pi 5—a device with the computing power of a smartphone, not a server. Most research papers achieve 5-10 FPS on similar hardware. We needed **20+ FPS** for smooth real-time operation.

How did we do it? Through systematic optimization. I'll hand it to Speaker 2 to explain our technical approach."

---

## 2. TECHNICAL ARCHITECTURE & OPTIMIZATION (Speaker 2 - Phat)
**Time:** 1:30 - 3:30

**(Slide: Architecture Diagram)**

**Speaker 2:**
"Thank you. Our system runs three AI models concurrently:
1. **Face Detection** - Finds faces in the video feed
2. **Face Recognition** - Identifies who they are
3. **Face Mesh** - Draws the 3D wireframe overlay

**(Slide: Performance Journey - 3 FPS → 20 FPS)**

**Speaker 2:**
"Initially, we achieved only **3 FPS**—completely unusable. Here's how we fixed it:

**Optimization 1: Detector Replacement**
We replaced YOLO (550ms per frame) with Google MediaPipe (10ms per frame). That's a **50x speedup** just by choosing the right model for our hardware.

**Optimization 2: ROI Cropping**
Instead of processing the entire 640x480 frame for the Face Mesh, we crop to just the detected face region. This reduced mesh processing time by 80%.

**Optimization 3: Threading Architecture**
We moved the heavy face recognition (150ms) to a background thread. The main thread handles visualization at full speed while recognition runs in parallel.

**(Slide: Final Performance Metrics)**

**Speaker 2:**
"The result? **18-20 FPS** with all three models running. For comparison, commercial systems like Ring use cloud processing—we do everything on-device.

Now, Speaker 3 will explain the AI models and a critical feature we added."

---

## 3. AI MODELS & IDENTITY MANAGEMENT (Speaker 3 - Tien)
**Time:** 3:30 - 5:00

**(Slide: Face Recognition Pipeline)**

**Speaker 3:**
"For face recognition, we use **ArcFace**—a state-of-the-art model trained on 5.8 million faces. It converts each face into a 512-dimensional 'fingerprint' that we compare against our database.

**(Slide: Custom Training Experiment)**

**Speaker 3:**
"We also experimented with training our own model from scratch on Kaggle. We collected 100 photos, trained for 50 epochs, and... it failed. Our custom model achieved only 65% accuracy compared to the pre-trained model's 95%.

**The lesson?** In deep learning, **data is king**. You can't beat a model trained on millions of images with just 100 photos. This was an educational success—we learned when NOT to reinvent the wheel.

**(Slide: Identity Deduplication)**

**Speaker 3:**
"One critical feature we added: **Identity Deduplication**. 

Imagine two people in frame, and both get recognized as 'John' because they look similar. Our system detects this conflict and keeps only the highest confidence match. The other person becomes 'Unknown'. This prevents false positives in multi-person scenarios.

**(Slide: Data Collection Tool)**

**Speaker 3:**
"We also built `add_user.py`—a tool that captures 150 photos in 10 seconds for training. Just run the script, rotate your head, and you're registered. This makes onboarding new family members effortless.

Now, Speaker 4 will show you the live demo."

---

## 4. LIVE DEMO & CONCLUSION (Speaker 4 - [Name 4])
**Time:** 5:00 - 7:00

**(Switch to Live Camera Feed)**

**Speaker 4:**
"Let's see it in action.

**(Point to screen)**

As you can see, the system is running right now. Notice:
- **The cyan wireframe** tracking my face in 3D—that's the Face Mesh
- **My name and confidence score** displayed in real-time
- **The FPS counter** showing 19-20 frames per second

**(Speaker 2 walks into frame)**

**Speaker 4:**
"Now Speaker 2 is entering. The system instantly detects and recognizes him as 'Phat' with 85% confidence. Both of us are tracked simultaneously without confusion.

**(Stranger enters OR cover face)**

**Speaker 4:**
"If I cover my face or a stranger appears, watch what happens..."

**(System shows "Unknown" in red)**

**Speaker 4:**
"The box turns red, labeled 'Unknown', and within 10 seconds, our Telegram bot sends an alert with a photo to the home owner's phone.

**(Show Telegram notification on phone)**

**Speaker 4:**
"Here's the alert. In a real deployment, this would notify you immediately if someone unauthorized enters your home.

**(Slide: Conclusion)**

**Speaker 4:**
"In conclusion, we built a production-ready security system that:
- Runs **three AI models** at 20 FPS on edge hardware
- Features **intelligent identity management** with deduplication
- Provides **instant alerts** via Telegram
- Looks like something from a sci-fi movie

**Future work** includes integrating the Hailo-8L NPU for 60+ FPS and adding cloud storage for recorded footage.

Thank you for your attention. We're happy to answer questions."

---

## Q&A PREPARATION

**Q: Why not use cloud processing like Ring or Nest?**
*A:* "Privacy and latency. Cloud processing requires uploading video to external servers, which raises privacy concerns. Our on-device processing is instant and keeps all data local."

**Q: What happens if the WiFi goes down?**
*A:* "The system continues to function locally—detection and recognition still work. Only the Telegram alerts require internet. We could add local storage as a backup."

**Q: How accurate is the face recognition?**
*A:* "With proper lighting and frontal faces, we achieve 95%+ accuracy using the pre-trained ArcFace model. The system requires at least 0.55 confidence (55% match) to identify someone, preventing false positives."

**Q: Can it handle multiple people?**
*A:* "Yes, we've tested with up to 3 people simultaneously. FPS drops to ~11 with 3 faces (due to 3x mesh rendering), but recognition remains accurate. The identity deduplication ensures no duplicate names."

**Q: Why Raspberry Pi instead of a more powerful device?**
*A:* "Cost and accessibility. A Raspberry Pi 5 costs $60 compared to $500+ for edge AI servers. We wanted to prove that sophisticated AI can run on affordable hardware with proper optimization."

**Q: What was the biggest technical challenge?**
*A:* "Achieving real-time performance. We went through three major optimization iterations (detector replacement, ROI cropping, threading) to go from 3 FPS to 20 FPS. Each required rethinking our approach rather than just tweaking parameters."

---

## DEMO BACKUP PLAN

**If WiFi fails:**
- Run demo using pre-recorded video: `python main.py --video demo_recording.mp4`
- Show Telegram alerts from previous test runs

**If camera fails:**
- Use laptop webcam: Modify `camera.py` to use `cv2.VideoCapture(0)`
- Show screenshots of working system

**If recognition fails:**
- Re-run `python tools/process_database.py` before presentation
- Have backup photos showing successful recognition

---

## TIMING BREAKDOWN
- Introduction: 1.5 min
- Architecture: 2 min
- AI Models: 1.5 min
- Demo: 2 min
- **Total:** 7 minutes (leaves 3 min for Q&A in 10-min slot)
