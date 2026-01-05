# FINAL PRESENTATION SCRIPT (4-Person Team)
**Project:** AI-Powered Home Security System (Raspberry Pi 5)
**Duration:** ~5-7 Minutes

---

## 1. INTRODUCTION (Speaker 1)
**Time:** 0:00 - 1:30

**(Slide: Title Card with "Iron Man" HUD Visual)**

**Speaker 1:**
"Good morning everyone. We are Team VacTon.
Home security today is boring. Most cameras just record footage. They don't *act*. They don't *inform*. And they definitely don't look cool.

Our goal was to build a security system that doesn't just watch, but **understands**. A system that knows *who* is at the door, tracks their movements in real-time, and provides a futuristic, military-grade User Interface."

**(Slide: Project Goals - Smart, Fast, Beautiful)**

**Speaker 1:**
"We set out to solve three problems:
1.  **Latency:** It has to be real-time.
2.  **Accuracy:** It has to know 'Friend' from 'Foe'.
3.  **Experience:** It needs to feel like advanced tech, not a cheap webcam."

"To do this, we maximized the power of the **Raspberry Pi 5**, moving beyond simple motion detection to full biometric analysis. I'll pass it to Speaker 2 to explain how we built the hardware foundation."

---

## 2. THE HARDWARE & OPTIMIZATION (Speaker 2)
**Time:** 1:30 - 3:00

**(Slide: Hardware Stack - Pi 5, Hailo (Optional), Camera)**

**Speaker 2:**
"Thank you.
At the core, we are using the **Raspberry Pi 5**. It’s a beast, but AI is heavy.
We faced a major challenge immediately: Running **Face Detection**, **Face Recognition**, AND **3D Face Mesh** simultaneously usually kills the frame rate.

**(Slide: The 'Optimization' - ROI Strategy)**

**Speaker 2:**
"Initially, our FPS dropped to single digits. We fixed this with a technique called **Region-of-Interest (ROI) Optimization**.
Instead of analyzing the whole 1080p frame for the Face Mesh, we implemented a smart pipeline:
1.  **YOLOv8** finds the face (The bounding box).
2.  We **Crop** just that tiny square.
3.  We run the heavy **Face Mesh** only on that crop.
4.  We project the points back to the main screen.

This jumped our performance from **4 FPS to over 20 FPS**, making the 'Iron Man' HUD possible on an edge device."

---

## 3. THE INTELLIGENCE & R&D (Speaker 3)
**Time:** 3:00 - 4:30

**(Slide: The 'Iron Man' HUD)**

**Speaker 3:**
"Now, let's talk about the AI.
We didn't just want a box around a face. We wanted meaningful data.
We integrated **Google MediaPipe**, which tracks **468 distinct landmarks** on the face. That’s the blue wireframe you see. It tracks blinking, smiling, and head rotation."

**(Slide: The Deep Learning Experiment)**

**Speaker 3:**
"We also conducted a major R&D experiment. We asked: *'Can we train a custom model better than Google?'*
We collected 100 photos of ourselves and built a **Custom Training Pipeline** on Kaggle, training a model from scratch.

The Result? **Rookie mistake.**
We learned the hard choice of 'Data Scarcity'. Our custom model was good, but it couldn't beat a pre-trained model trained on **5.8 million images**.
So, for the final product, we made the engineering decision to stick with the **Pre-trained ArcFace Model** for maximum security, while keeping our custom visualization layer. This gave us the best of both worlds: Professional Integrity and Educational Experience."

---

## 4. LIVE DEMO & CONCLUSION (Speaker 4)
**Time:** 4:30 - 6:00

**(Slide: Live Demo)**

**Speaker 4:**
"Enough talk. Let's see it in action."

*(Switch to Live Feed)*

**Speaker 4:**
"As you can see, the system instantly locks onto my face.
*   **The Blue Mesh:** That's the ROI optimization Speaker 2 talked about. Note how it follows my head rotation perfectly.
*   **The Name Tag:** 'Speaker 4 (0.85)'. That's the ArcFace model verifying my identity.
*   **Stranger Alert:** If I cover my face or if a stranger enters, the HUD turns **Red** and a Telegram alert is sent instantly."

*(Demo concludes)*

**Speaker 4:**
"In conclusion, we built a system that is robust, fast, and visually effectively. We pushed the Raspberry Pi 5 to its limit, balancing heavy AI workloads with real-time performance.
We are Team VacTon. Thank you for listening."

---
**Q&A Preparation:**
*   *Q: Why not use Hailo for everything?*
    *   A: "We use it for detection, but some custom layers like Face Mesh run efficiently on CPU with our ROI trick."
*   *Q: Why did the custom training fail?*
    *   A: "Small dataset = Overfitting. We needed thousands of pictures to beat the pre-trained weights."
