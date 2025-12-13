---
layout: default
title: "Mind the Sensors"
---

# **Mind the Sensors: Interactive Consent for Smart Environments**

*A multi-modal system enabling transparent, interactive user consent for data collection in smart spaces.*

![System Architecture](./assets/img/system_architecture_placeholder.png)
<sub>*(Recommended: Insert the System Architecture diagram from Slide 3 here)*</sub>

---

## 👥 **Team**

- Junyu Liu
- Maoqi Xu
- Jacob Yang
- **Mentor:** Brian Wang
- **Professor:** Prof. Mani Srivastava

---

## 📝 **Abstract**

Smart environments often suffer from a lack of transparency, leaving visitors feeling watched rather than served. To address this, we developed "Mind the Sensors," an interactive system that allows users to easily grant or deny data collection consent via natural modalities like voice, camera, and web interfaces. Our approach utilizes a Raspberry Pi-based embedded platform running local facial and voice recognition pipelines to ensure privacy. Key results demonstrate high user trust and ease of use across scenarios, with keyword spotting achieving near 100% accuracy. This project provides building owners and developers with reusable patterns to deploy privacy-respecting smart spaces.

---

## 📑 **Slides**

- [Midterm Presentation Slides](./assets/Project11_Midterm_Presentations.pdf)
- [Final Presentation Slides](./assets/Project11_FinalPresentations.pdf)

---

## 🎛️ **Media**

- *(Optional: Add link to demo video if available)*

---

# **1. Introduction**

As smart environments become ubiquitous, the need for clear, accessible privacy controls becomes critical. This project introduces a system that empowers users to control their digital footprint in physical spaces.

### **1.1 Motivation & Objective**
The primary objective is to let people walking into a smart room easily say "yes" or "no" to what sensors collect about them. We aim to create simple, clear consent prompts that work with available modalities—whether that is a user's phone, a camera, or their voice—ensuring the space remembers their choice so they are not asked twice.

### **1.2 State of the Art & Its Limitations**
Current smart space implementations often fail to inform visitors, leading to a feeling of being "watched". Furthermore, visitors with disabilities or those without electronic devices at hand are often excluded from digital consent flows.

### **1.3 Novelty & Rationale**
Our system introduces a novel multi-modal consent framework. It supports four concrete entry setups: camera-web based, camera-voice based, voice-only based, and voice-website based. This flexibility ensures inclusivity and robust interaction regardless of the user's hardware status.

### **1.4 Potential Impact**
* **Visitors:** Feel informed and in control.
* **Building Owners:** Gain a cleaner, trusted way to run smart spaces that respect privacy rules.
* **Developers:** Gain reusable patterns for adding consent to real physical spaces.

### **1.5 Challenges**
* **Privacy:** Ensuring biometric data (face/voice) is processed locally without external cloud dependencies.
* **Latency:** Achieving real-time processing on embedded hardware (Raspberry Pi).
* **Accuracy:** Balancing strict matching thresholds against usability to prevent false positives.

### **1.6 Metrics of Success**
We evaluated the system based on:
1.  **Usability:** Clarity, Ease of Use, and Trust ratings from users.
2.  **Performance:** Success Rate, False Match Rate (FMR), and False Non-match Rate (FNMR).

---

# **2. Related Work**

Our work builds upon established libraries in computer vision and speech processing, integrating them into a novel privacy framework:

* **Facial Recognition:** We utilize `face_recognition` and `opencv`, leveraging dlib's ResNet models for 128-d face embeddings.
* **Speaker Verification:** We employ **Resemblyzer**, derived from Google's GE2E (Generalized End-to-End Loss) model, for extracting voiceprints.
* **Speech Recognition:** We use **Vosk** for offline, lightweight speech-to-text processing.
* **Privacy Compliance:** We integrate **iubenda** for standardized cookie and privacy policy management.

---

# **3. Technical Approach**

We designed a modular system centered around a Raspberry Pi that orchestrates sensor input, database management, and user feedback.

### **3.1 System Architecture**
The system comprises three main interaction loops:
1.  **Camera:** Captures face IDs to query consent status.
2.  **Microphone:** Captures voice commands and biometric voiceprints.
3.  **Server/Web:** Manages the database and serves consent pages to user phones.

### **3.2 Data Pipeline & Database**
We use a lightweight **SQLite** database to store identity and consent status locally.
* **Table `users`:** Stores `user_id`, `name`, and `permission` (0=first seen, 1=consent, 2=no consent).
* **Table `user_faces`:** Stores 128-d face embeddings as BLOBs.
* **Table `user_voices`:** Stores voice recognition embeddings.

### **3.3 Facial Recognition Logic**
* **Input:** `Picamera2` via libcamera for direct hardware access.
* **Preprocessing:** 4x downscaling using `cv_scaler` to maintain real-time FPS.
* **Logic:** Faces are converted to 128-d vectors. We calculate L2 distance against SQLite records with a strict match threshold of **0.38**.
* **Temporal Clustering:** To handle auto-enrollment, we buffer "unknown" encodings and require 5 consistent frames within a 3-second window before generating a new Visitor profile.

### **3.4 Voice Interaction & NLP Module**
* **Wake Word:** Uses Vosk (offline ASR) to detect the wake phrase "Apple".
* **Biometrics:** Converts 3-second audio clips into 256-dimensional embeddings using Resemblyzer. Matching uses Cosine Similarity with a threshold of **0.75**.
* **Intent Recognition:** The NLP module analyzes audio for positive keywords ("yes", "sure") or negative keywords ("no", "nope") within a 5-second window.

### **3.5 Server & Privacy Integration**
The Flask application handles API polling and web serving. Crucially, we integrated **iubenda** to embed standardized cookie and privacy policy solutions directly into the templates, ensuring compliance with US State Laws.

---

# **4. Evaluation & Results**

We conducted usability testing with users across different interaction modalities.

### **4.1 Usability Metrics**

**Camera-Based Scenarios:** Users rated "Trust" consistently high (5/5). Success rates were generally high (3/3), though some users experienced minor friction (2/3).

| Metric | Voice Interaction | Website Interaction |
| :--- | :--- | :--- |
| **Clarity** | 3.0 - 4.0 | 4.0 - 5.0 |
| **Ease of Use** | 2.0 - 3.0 | 3.0 - 4.0 |
| **Trust** | **5.0** | **5.0** |

**Voice-Based Scenarios:** Voice interaction showed improved clarity scores compared to camera scenarios.

| Metric | Voice Interaction | Website Interaction |
| :--- | :--- | :--- |
| **Clarity** | 4.0 - 5.0 | 5.0 |
| **Ease of Use** | 4.0 | 4.0 - 5.0 |
| **Trust** | 4.0 - 5.0 | 4.0 - 5.0 |

### **4.2 System Performance**
* **Facial Recognition:**
    * False Non-match Rate (FNMR): **0** (Official dlib rate is ~0.2%).
    * False Match Rate (FMR): **6/20** (Note: Threshold was lowered to 0.38 to prioritize detection).
* **Voice Recognition:**
    * Keyword Spotting Accuracy: **~100%** (No mismatches for yes/no commands).
    * False Match Rate (FMR): **3/20**.

---

# **5. Discussion & Conclusions**

Our system successfully demonstrated that complex privacy consent flows can be simplified into natural interactions.

* **What worked:** The multi-modal approach (voice and camera) provided robust alternatives for users. Trust scores were consistently high (4.0-5.0), validating our privacy-first local processing approach. Keyword spotting was highly reliable.
* **Limitations:** The Facial Recognition FMR was higher than ideal (6/20) due to the aggressive thresholding required for the demo environment.
* **Future Work:** We would refine the biometric thresholds to reduce false matches and explore more advanced "liveness" detection to prevent spoofing.

---

# **6. References**

1.  **Dlib C++ Library** (Facial Recognition underlying model)
2.  **OpenCV** (Image processing)
3.  **Vosk** (Offline Speech Recognition)
4.  **Resemblyzer** (Voice Verification)
5.  **Flask** (Web Framework)
6.  **Iubenda** (Privacy/Cookie Solution)

---

# **7. Supplementary Material**

## **7.a. Datasets**
* **User Encodings:** Collected locally during the demo phase. Stored as 128-d (face) and 256-d (voice) vectors in SQLite.

## **7.b. Software**
* **Facial Recognition:** `face_recognition` + `opencv.cv2`
* **Voice Processing:** `sounddevice` (I/O), `soundfile` (output)
* **Database:** SQLite
