# 🛡️ DeepGuard  
### Real-Time Deepfake Detection for Images, Videos & Live Screen Content

DeepGuard is a **real-time deepfake detection system** capable of analyzing **images, videos, webcam feeds, and live on-screen content** (including social media and websites).  
It is designed as a **lightweight, platform-agnostic AI safety engine** focused on real-world usability rather than benchmark-only performance.

---

## 🚀 Key Features

- 🖼️ Image deepfake detection  
- 🎞️ Video deepfake detection  
- 📷 Live webcam analysis  
- 🖥️ Live screen / website / social media monitoring  
- 🧠 Face-based deepfake classification (MesoNet)  
- 🧩 Modular, extensible architecture  
- 🔍 Explainability layer (Gemini-ready with robust fallback)

---

## 🧠 System Overview

DeepGuard follows a **modular, pipeline-based architecture**:

1. **Face Detection**  
   Faces are detected using **MTCNN** from:
   - images  
   - videos  
   - webcam streams  
   - live screen frames  

2. **Deepfake Classification**  
   Each detected face is processed by a **pretrained MesoNet CNN**, producing a deepfake probability score.

3. **Temporal Aggregation**  
   For videos and live streams, predictions are aggregated across frames to reduce noise and improve stability.

4. **Explanation Layer**  
   Detection results are converted into **human-readable explanations** using a Gemini-compatible design with a deterministic fallback to ensure reliability.

---

## 📁 Project Structure

```text
deepguard/
├── core/        # Detection pipelines & logic
├── mesonet/     # Deepfake model architecture & pretrained weights
├── requirements.txt
├── README.md
└── .gitignore
````

Each pipeline can be run **independently**.

---

## ▶️ How to Run

### 1️⃣ Setup (Windows)

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

### 2️⃣ Run Pipelines

### 🖼️ Image

```bash
python -m core.image_pipeline --path face_0.jpg
```

or

```bash
python -m core.image_pipeline --image face_0.jpg
```

---

### 🎞️ Video

```bash
python -m core.video_pipeline --path "path/to/video.mp4"
```

or

```bash
python -m core.video_pipeline --video "path/to/video.mp4"
```

---

### 📷 Webcam

```bash
python -m core.live_pipeline
```

---

### 🖥️ Live Screen / Social Media

```bash
python -m core.screen_pipeline
```

---

## 🔐 Gemini Integration (Explainability Layer)

DeepGuard is designed with **Gemini API integration** for AI-based reasoning and explanation of detection results.

Due to current API access and model availability constraints:

* The system uses a **deterministic fallback**
* Gemini-compatible prompts, model discovery, and architecture are preserved

This ensures:

* robustness
* transparency
* production readiness

---

## 🧩 Use Cases

* Social media deepfake monitoring
* Content moderation pipelines
* Media forensics & verification
* Browser or application integration
* AI safety and trust research

---

## 🎬 Note on Movie & Cinematic Content

DeepGuard may occasionally flag **movie scenes or cinematic footage** as potential deepfakes.
This is expected behavior due to:

* heavy visual effects (VFX)
* CGI-based face enhancement
* cinematic color grading
* compression artifacts

The system is **intentionally conservative**, prioritizing safety over permissiveness.
This behavior is acceptable and expected in moderation-focused applications.

---

## ⚠️ Disclaimer

DeepGuard is a **research prototype**.
Predictions may be affected by:

* video quality
* lighting conditions
* compression
* artistic or cinematic effects

The system should be used as a **decision-support tool**, not as an absolute authority.

---

## 📜 License

For research and educational use.

---

## 🌱 Future Work

* Integration with stronger temporal models
* Transformer-based deepfake classifiers
* Mobile and browser deployment
* Multi-modal reasoning using Gemini

---

## ✨ Why This Project Matters

DeepGuard focuses on **real-world deployability** rather than benchmark-only performance.
By enabling **live, on-device deepfake detection**, it addresses a growing need for scalable AI safety tools in modern digital platforms.

```
