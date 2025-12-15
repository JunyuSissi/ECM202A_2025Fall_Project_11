---
layout: default
title: "Final Report"
---

# Table of Contents
* [Introduction](#1-introduction)
* [Related Work](#2-related-work)
* [Technical Approach](#3-technical-approach)
* [Evaluation and Results](#4-evaluation--results)
* [Discussion and Conclusions](#5-discussion--conclusions)
* [References](#6-references)
* [Suuplementary Material](#7-supplementary-material)

# **1. Introduction**
Smart envrionments increasingly rely on sensors such as camera and microphones to provide context-aware services, yet these systems often lack clear and accessfible mechanisms for obtaining user consent.As a result, visitors may feel surveilled rather than served, particularly when sensing occurs without explicit notice or meaningful choice. 

This project introduces *Mind the Sensors*, a multi-modal consent system that empowers users to control their digital footprint in physical spaces through natural interactions. Inpired by cookie consent banner on websites, which inform users of data collection and offerimmediate accept/deny choices with optional granular controls, our system aims to establish an equivalent, intuitive consent mechanism for smart envrionments.

By combining local biometric identification with verbal consent prompts and an optional web interface, our system enables users to express privacy preferences at the moment of data collection. All processing is performed locally on a Raspberry Pi, ensuring low latency and avoiding cloud-based data exposure. Through this work, we explore how familiar web privacy interaction patterns can be translated into embodied, real-world settings.

### **1.1 Motivation & Context**
Current smart space implementations often fail to inform visitors, leading to a feeling of being "watched" or "surveilled." Furthermore, standard consent mechanisms (like website pop-ups) do not translate well to physical spaces. Visitors with disabilities or those without electronic devices at hand are often excluded from digital consent flows.

The primary objective is to let people walking into a smart room easily say "yes" or "no" to what sensors collect about them. We aim to:
1.  **Simplify Consent:** Create prompts that work with available modalities—phone, camera, or voice.
2.  **Ensure Persistence:** Ensure the space remembers a user's choice to avoid repetitive prompts.
3.  **Preserve Privacy:** Process biometric data locally without cloud dependency.

### **1.2 State of the Art & Its Limitations**
Prior research in ubiquitous computing emphasizes the importance of transparency, notice, and user consent in smart envrionnments. Langheinrich’s Privacy Awareness System (pawS) [[2](#6-references)] established foundational principles such as notice, choice, and accountability, arguing that users should be informed and able to express preferences at the time of data collection. However, these systems often rely on abstract policy descriptions or centralized preference management, which are difficult to access during real-time physical interactions.

More recent work such as Peekaboo focuses on privacy-by-design by preprocessing sensor data locally to reduce cloud exposure [[1](#6-references)]. While effective at minimizing downstream privacy risks, these approaches do not provide explicit mechanisms for obtaining user consent at the moment data is captured. Similarly, stakeholder-focused access-control systems such as TEO [[3](#6-references)] address power asymmetries in shared spaces through ephemeral ownership models, but introduce deployment complexity and do not prioritize lightweight, user-facing consent interactions.

In contrast, on the web, cookie consent banners have become a standardized mechanism for providing notice, collecting user consent, and offering optional granular privacy controls under regulations such as GDPR and CCPA [[12](#6-references)]. Despite their limitations, cookie banners establish a widely understood mental model for privacy decision-making. No equivalent, standardized interaction exists for physical smart environments, leaving a gap between users’ expectations formed online and consent mechanisms available in the real world.

### **1.3 Novelty & Rationale**
The novelty of Mind the Sensors lies in translating the web-based cookie consent paradigm into physical smart environments through multimodal, in-situ interaction. Inspired by cookie consent banners, our system provides:
1. immediate notice upon entry or identification 
2. a simple binary consent decision via voice
3. optional granular controls through a web interface.

Unlike traditional web consent flows, our design does not assume the presence of screens or keyboards. Instead, we support two complementary scenarios: a camera-assisted multimodal flow and a fully camera-free, voice-only flow. This design improves accessibility for users with visual or hearing impairments and enables deployment in diverse public and commercial settings.

All biometric processing and permission storage are performed locally on a Raspberry Pi, aligning with privacy-by-design principles while keeping interaction latency low. By combining familiar consent metaphors with edge-based processing and embodied interaction, our system bridges the gap between online privacy practices and physical smart spaces.

### **1.4 Potential Impact**
Mind the Sensors demonstrates how familiar web privacy interaction patterns can be adapted to physical environments, lowering the cognitive burden of consent in smart spaces. For users, the system provides intuitive, accessible, and immediate control over data collection. For building operators and developers, it offers a low-cost, deployable framework that improves transparency without relying on cloud infrastructure.

By supporting voice-based and camera-free interaction and keeping all sensitive data local, the system advances accessibility and privacy-by-design in public smart environments. More broadly, this work contributes reusable design patterns for consent-aware sensing systems and highlights a path toward standardized privacy interactions beyond the web.

### **1.5 Challenges**

Designing an interactive consent system for physical smart environments presents several challenges.

First, **accuracy versus usability trade-offs** arise in biometric identification. Lowering recognition thresholds improves responsiveness and reduces interaction friction, but increases the risk of false matches, particularly under variable lighting and acoustic conditions. This trade-off was evident in our facial recognition pipeline, where fast detection led to a higher false match rate.

Second, **environmental variability** significantly impacts system performance. Changes in lighting, camera angle, background noise, and user speaking style affect both face and voice recognition accuracy, making robust identification difficult in uncontrolled public settings.

Third, **accessibility and modality balance** pose design challenges. While voice-based interaction improves accessibility for users without screens or with visual impairments, it may be less suitable in noisy environments or for users with speech impairments. Supporting multiple modalities without overwhelming users requires careful orchestration.

Finally, **user trust and expectation management** remains a challenge. Even when data is processed locally, users may perceive biometric sensing as invasive. Designing prompts and feedback that clearly communicate system behavior and limitations is essential for maintaining trust.

### **1.6 Metrics of Success**

The success of *Mind the Sensors* is evaluated along two primary dimensions: usability and technical performance.

From a usability perspective, success is measured through user-reported ratings of clarity, ease of use, and trust across both interaction scenarios. High trust scores (4.0–5.0) and improved clarity in voice-based interactions indicate that users understood the consent prompts and felt comfortable expressing their choices.

From a technical perspective, success is measured using biometric performance metrics, including false match rate (FMR), false non-match rate (FNMR), and keyword spotting accuracy. Near-perfect keyword detection and zero false non-matches in facial recognition demonstrate reliable interaction triggering, while observed false matches highlight areas for future improvement rather than system failure.

Additionally, system responsiveness and local-only processing serve as implicit success metrics, validating that interactive consent can be achieved on low-power embedded hardware without cloud dependency.

---

# **2. Related Work**

Our work builds upon established research in ubiquitous computing, privacy-by-design, and biometric authentication.

### **2.1 Privacy in Smart Spaces**
* **Langheinrich (2001)**[[2](#6-references)] established the principles of "Privacy Awareness Systems" in ubiquitous computing, emphasizing the need for notice and consent.
* **Schaub et al. (2015)**[[4](#6-references)] proposed a design space for effective privacy notices, arguing that notices must be context-aware and delivered at the right time. Our system implements this by triggering consent requests *only* upon entry or identification.

### **2.2 Biometric Identification**
* **Face Recognition:** We utilize **dlib**'s ResNet models, which achieve 99.38% accuracy on the LFW benchmark. Unlike cloud-based solutions (e.g., AWS Rekognition), our implementation focuses on **edge-based** inference using `face_recognition` [[7](#6-references)] and `opencv` [[11](#6-references)] to keep data local.
* **Speaker Verification:** We employ **Resemblyzer** [[8](#6-references)], derived from Google's **GE2E (Generalized End-to-End Loss)** model. This allows for real-time voice embedding generation on low-power devices, a significant improvement over older GMM-UBM models.

### **2.3 Legal Compliance Tools**
* **Iubenda:** While mostly used for web compliance, we integrate **iubenda**'s API to manage "Cookie Solutions" and legal policy generation [[5](#6-references)], bridging the gap between physical sensor data and GDPR/CCPA digital compliance requirements.

### **2.4 Web-Based Consent and Cookie Banners**
Cookie consent banners are the dominant mechanism for communicating data collection practices and obtaining user consent on websites, particularly under GDPR and CCPA regulations. These interfaces typically provide layered consent, combining immediate accept/deny actions with expandable options for granular control. Prior work on privacy notices emphasizes that effective consent mechanisms must be timely, contextual, and easy to understand.

Our system draws direct inspiration from this interaction model, adapting it to physical environments where conventional web interfaces are unavailable. By offering a fast verbal consent path alongside a detailed web interface accessed via NFC, Mind the Sensors mirrors the structure of cookie consent dialogs while respecting the constraints of embodied, real-world interaction.

---

# **3. Technical Approach**

We designed a modular system centered around a Raspberry Pi that orchestrates sensor input, database management, and user feedback.

### **3.1 System Architecture**
The system is built on an embedded platform using a Raspberry Pi, which acts as the central coordinator. It runs a Flask-based web server [[9](#6-references)] to handle local data processing and user interface rendering. To accommodate different user needs, we designed two distinct architectural flows.

#### **3.1.1 Voice-Recognition Architecture**
This flow focuses on identifying users and capturing consent primarily through audio channels, ideal for hands-free interaction.

![Voice Recognition System Design]({{ site.baseurl }}/assets/img/Voice_Recognition_based_system_design.png)


*Figure 2: Architecture for the voice-based interaction flow. The Microphone captures verbal IDs, processed by the Voice Recognition module, and validated against the Database.*

#### **3.1.2 Facial-Recognition Architecture**
This flow uses a multimodal approach, combining visual identification with verbal consent for higher accuracy.

![Facial Recognition System Design]({{ site.baseurl }}/assets/img/Facial_Recognition_based_system_design.png)


*Figure 3: Architecture for the camera and voice hybrid flow. The Camera triggers the identification, while the NLP module processes verbal consent.*

### **3.2 Data Pipeline & Database**
We use a lightweight **SQLite** database [[6](#6-references)] to store identity and consent status locally. This ensures no sensitive biometric data leaves the device.
* **Table `users`:** Stores `user_id`, `name`, and `permission` (0=first seen, 1=consent, 2=no consent).
* **Table `user_faces`:** Stores 128-d face embeddings as BLOBs.
* **Table `user_voices`:** Stores voice recognition embeddings.

### **3.3 Algorithmic Details**
* **Facial Logic:** Input via `Picamera2` is downscaled 4x for speed. We calculate L2 distance against SQLite records with a strict match threshold of **0.38**. To handle auto-enrollment, we use **Temporal Clustering**: buffering "unknown" encodings and requiring 5 consistent frames within a 3-second window.
* **Voice Logic:** Using **Vosk** [[10](#6-references)] for wake-word detection ("Apple") and **Resemblyzer** [[8](#6-references)] for biometric verification. Matching uses Cosine Similarity with a threshold of **0.75**.

---

# **4. Evaluation & Results**

We evaluated the system based on **Usability** (User trust/clarity) and **Technical Performance** (Accuracy/Latency).

### **4.1 Usability Testing**
We conducted user studies across two main scenarios. Users rated their experience on a scale of 1-5.

| Metric | Camera Scenario (Avg) | Voice Scenario (Avg) | Notes |
| :--- | :--- | :--- | :--- |
| **Clarity** | 3.5 | **4.5** | Voice prompts were perceived as clearer instructions. |
| **Ease of Use** | 2.5 | **4.0** | Camera positioning caused friction for some users. |
| **Trust** | **5.0** | 4.5 | Visual feedback (seeing the camera) instilled high trust. |

> *Observation:* While the Camera scenario had lower "Ease of Use" due to positioning requirements, it achieved the highest "Trust" score, suggesting users prefer seeing the sensor that is tracking them.

### **4.2 Technical Performance**

#### **Facial Recognition Accuracy**
We tested the system with 20 trial interactions.
* **False Non-match Rate (FNMR):** **0%**. The system never failed to recognize a registered user.
* **False Match Rate (FMR):** **30% (6/20)**.
    * *Analysis:* The FMR was higher than the theoretical baseline because we lowered the threshold to **0.38** to ensure fast detection in varying lighting conditions.

#### **Voice Recognition Accuracy**
* **Keyword Spotting:** **~100%**. No mismatches for simple "yes/no" commands.
* **Biometric FMR:** **15% (3/20)**.

---

# **5. Discussion & Conclusions**

Our system successfully demonstrated that complex privacy consent flows can be simplified into natural interactions.

* **Successes:** The multi-modal approach provided robust alternatives. Trust scores were consistently high (4.0-5.0), validating our privacy-first local processing approach.
* **Limitations:** The Facial Recognition FMR (30%) is too high for high-security contexts but acceptable for a consent prototype. Lighting conditions significantly impacted the camera's ability to maintain a lock.
* **Future Work:**
    1.  **Liveness Detection:** Implement blink-detection to prevent photo-spoofing.
    2.  **Adaptive Thresholds:** Dynamically adjust the L2 distance threshold based on ambient lighting.

---

# **6. References**
[1] H. Jin, G. Liu, D. Hwang, S. Kumar, Y. Agarwal, and J. I. Hong, "Peekaboo: A hub-based approach to enable transparency in data processing within smart homes," in *2022 IEEE Symposium on Security and Privacy (SP)*, San Francisco, CA, USA, 2022, pp. 303-320. [Online]. Available: [link](https://arxiv.org/pdf/2204.04540).

[2] Marc Langheinrich. 2002. A Privacy Awareness System for Ubiquitous Computing Environments. In Proceedings of the 4th international conference on Ubiquitous Computing (UbiComp '02). Springer-Verlag, Berlin, Heidelberg, 237–245.

[3] Zhang, Han et al. “TEO: ephemeral ownership for IoT devices to provide granular data control.” Proceedings of the 20th Annual International Conference on Mobile Systems, Applications and Services (2022): n. pag.

[4] Florian Schaub, Rebecca Balebako, Adam L. Durity, and Lorrie Faith Cranor. 2015. A design space for effective privacy notices. In Proceedings of the Eleventh USENIX Conference on Usable Privacy and Security (SOUPS '15). USENIX Association, USA, 1–17.

[5] Iubenda, "GDPR Cookie Consent Cheatsheet," *Iubenda Help*, [Online]. Available: [link](https://www.iubenda.com/en/help/23672-gdpr-cookie-consent-cheatsheet).

[6] D. R. Hipp, "SQLite," *SQLite Documentation*, 2025. [Online]. Available: [link](https://www.sqlite.org/index.html).

[7] A. Geitgey, "Face Recognition," *GitHub Repository*, 2018. [Online]. Available: [link](https://github.com/ageitgey/face_recognition).

[8] CorentinJ, "Resemblyzer: A Deep Learning Voice Encoder," *GitHub Repository*, 2019. [Online]. Available: [link](https://github.com/resemble-ai/Resemblyzer).

[9] Pallets Projects, "Flask: User's Guide," *Flask Documentation*, 2024. [Online]. Available: [link](https://flask.palletsprojects.com/).

[10] Alpha Cephei, "Vosk Offline Speech Recognition API," [Online]. Available: [link](https://alphacephei.com/vosk/).

[11] OpenCV Team, "OpenCV (Open Source Computer Vision Library)," [Online]. Available: [link](https://opencv.org/).

[12] European Data Protection Board. Guidelines on consent and cookies under GDPR.

---

# **7. Supplementary Material**

To encourage further development, we have documented our software stack and datasets.

### **7.a. Datasets**
* **User Encodings:** Collected locally during the demo phase. Stored as 128-d (face) and 256-d (voice) vectors.

### **7.b. Software**
* **Facial Recognition:** `face_recognition`, `opencv-python`
* **Voice Processing:** `sounddevice`, `vosk`, `resemblyzer`
* **Backend:** `Flask`, `SQLite3`
* **Compliance:** `Iubenda` API