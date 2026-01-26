<div align="center">

# 🛡️ DeepGuard

### Real-Time Deepfake Detection for Images, Videos & Live Screen Content

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Gemini API](https://img.shields.io/badge/Gemini-API%20Ready-4285F4.svg)](https://ai.google.dev/)

**🏆 Built for the [Gemini 3 Hackathon](https://gemini3.devpost.com/) by Google DeepMind**

[Features](#-key-features) • [Architecture](#️-architecture) • [Quick Start](#️-quick-start) • [Usage](#-usage) • [Gemini API](#-gemini-integration)

</div>

---

## 📖 Overview

DeepGuard is a **real-time deepfake detection system** capable of analyzing **images, videos, webcam feeds, and live on-screen content** (including social media and websites).

It is designed as a **lightweight, platform-agnostic AI safety engine** focused on real-world usability rather than benchmark-only performance.

---

## 🚀 Key Features

| Feature | Description |
|---------|-------------|
| 🖼️ **Image Detection** | Analyze static images for deepfake manipulation |
| 🎞️ **Video Detection** | Process video files with temporal aggregation |
| 📷 **Live Webcam** | Real-time detection from webcam feed |
| 🖥️ **Screen Capture** | Monitor any on-screen content (social media, websites) |
| 🧠 **MesoNet CNN** | Lightweight face-based deepfake classification |
| 🔍 **Explainability** | Gemini-powered human-readable explanations |
| 🧩 **Modular Design** | Each pipeline works independently |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DeepGuard System                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
         ▼                            ▼                            ▼
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│  INPUT SOURCES  │        │  INPUT SOURCES  │        │  INPUT SOURCES  │
│                 │        │                 │        │                 │
│  📷 Webcam      │        │  🖼️ Image       │        │  🖥️ Screen      │
│  🎞️ Video       │        │                 │        │    Capture      │
└────────┬────────┘        └────────┬────────┘        └────────┬────────┘
         │                          │                          │
         └──────────────────────────┼──────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      FACE DETECTION (MTCNN)   │
                    │                               │
                    │  • Multi-face detection       │
                    │  • Bounding box extraction    │
                    │  • Face cropping & alignment  │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   DEEPFAKE CLASSIFIER         │
                    │        (MesoNet CNN)          │
                    │                               │
                    │  Input: 256×256 RGB face      │
                    │  Output: Probability [0-1]    │
                    │                               │
                    │  ┌─────────────────────────┐  │
                    │  │ Conv2D → BatchNorm →    │  │
                    │  │ MaxPool (×4 blocks)     │  │
                    │  │ → Flatten → Dense →     │  │
                    │  │ Dropout → Sigmoid       │  │
                    │  └─────────────────────────┘  │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
        ┌───────────────────┐           ┌───────────────────┐
        │ SINGLE FRAME      │           │ TEMPORAL          │
        │ (Image Pipeline)  │           │ AGGREGATION       │
        │                   │           │ (Video/Live)      │
        │ Direct output     │           │                   │
        └─────────┬─────────┘           │ Avg across frames │
                  │                     └─────────┬─────────┘
                  │                               │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │     EXPLANATION LAYER         │
                  │        (Gemini API)           │
                  │                               │
                  │  • Human-readable reasoning   │
                  │  • Confidence interpretation  │
                  │  • Deterministic fallback     │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │          OUTPUT               │
                  │                               │
                  │  {                            │
                  │    "label": "deepfake|real",  │
                  │    "score": 0.0-1.0,          │
                  │    "confidence": "high|med|low"│
                  │    "explanation": "..."       │
                  │  }                            │
                  └───────────────────────────────┘
```

---

## 📁 Project Structure

```
deepguard/
├── core/                          # Detection pipelines & logic
│   ├── detector.py                # Core DeepfakeDetector class
│   ├── face_detector_mtcnn.py     # MTCNN face detection wrapper
│   ├── gemini_explainer.py        # Gemini API explanation layer
│   ├── image_pipeline.py          # Static image analysis
│   ├── video_pipeline.py          # Video file processing
│   ├── live_pipeline.py           # Webcam real-time detection
│   ├── screen_pipeline.py         # Screen capture monitoring
│   └── test_*.py                  # Unit tests
│
├── mesonet/                       # MesoNet model (WIFS 2018)
│   ├── classifiers.py             # Meso4, MesoInception4 architectures
│   ├── weights/                   # Pretrained model weights
│   │   ├── Meso4_DF.h5            # Deepfake detection weights
│   │   ├── Meso4_F2F.h5           # Face2Face detection weights
│   │   └── MesoInception_*.h5     # Inception variant weights
│   └── test_images/               # Sample test images
│
├── sample video/                  # Demo video for testing
├── requirements.txt               # Python dependencies
├── LICENSE                        # Apache 2.0 License
├── CONTRIBUTING.md                # Contribution guidelines
├── SECURITY.md                    # Security policy
└── README.md                      # This file
```

---

## ⚡️ Quick Start

### Prerequisites

- Python 3.10+
- pip
- Webcam (for live detection)

### Installation

```bash
# Clone the repository
git clone https://github.com/omtripathi52/deepguard.git
cd deepguard

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🎯 Usage

### 🖼️ Image Detection

```bash
python -m core.image_pipeline --path face_0.jpg
# or
python -m core.image_pipeline --image face_0.jpg
```

### 🎞️ Video Detection

```bash
python -m core.video_pipeline --path "path/to/video.mp4"
# or
python -m core.video_pipeline --video "path/to/video.mp4"
```

### 📷 Webcam (Real-Time)

```bash
python -m core.live_pipeline
```
> Press `q` to quit

### 🖥️ Screen Capture

```bash
python -m core.screen_pipeline
```
> Press `Ctrl+C` to stop

---

## 🔐 Gemini Integration

DeepGuard is designed with **Gemini API integration** for AI-based reasoning and explanation of detection results.

### How It Works

| Component | Description |
|-----------|-------------|
| **Prompt Design** | Structured prompts for detection reasoning |
| **Model Discovery** | Automatic Gemini model enumeration |
| **Fallback System** | Deterministic explanations when API unavailable |

### Configuration

```bash
# Set your Gemini API key (optional - fallback works without it)
export GEMINI_API_KEY="your_api_key_here"
```

### Explanation Output Example

```
The system flagged this video as a potential deepfake with 84% confidence.
This may be due to subtle facial inconsistencies, unnatural motion patterns,
or artifacts commonly introduced by synthetic media generation.
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **TensorFlow/Keras** | Deep learning framework |
| **MTCNN** | Multi-task face detection |
| **MesoNet** | Deepfake classification (WIFS 2018) |
| **OpenCV** | Image/video processing |
| **mss** | Cross-platform screen capture |
| **Google Gemini** | AI-powered explanations |

---

## 🧩 Use Cases

- ✅ Social media deepfake monitoring
- ✅ Content moderation pipelines
- ✅ Media forensics & verification
- ✅ Browser or application integration
- ✅ AI safety and trust research

---

## 🎬 Note on Movie & Cinematic Content

DeepGuard may occasionally flag **movie scenes or cinematic footage** as potential deepfakes due to:

- Heavy visual effects (VFX)
- CGI-based face enhancement
- Cinematic color grading
- Compression artifacts

> **This is expected behavior.** The system is intentionally conservative, prioritizing safety over permissiveness.

---

## ⚠️ Disclaimer

DeepGuard is a **research prototype**. Predictions may be affected by:

- Video quality
- Lighting conditions
- Compression
- Artistic or cinematic effects

**Use as a decision-support tool, not as an absolute authority.**

---

## 🌱 Future Work

- [ ] Integration with stronger temporal models
- [ ] Transformer-based deepfake classifiers
- [ ] Mobile and browser deployment
- [ ] Multi-modal reasoning using Gemini
- [ ] Real-time confidence calibration

---

## 📜 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **MesoNet** architecture by [Darius Afchar et al.](https://arxiv.org/abs/1809.00888) (WIFS 2018)
- **MTCNN** for face detection
- **Google Gemini** for explainability layer

---

## ✨ Why This Project Matters

DeepGuard focuses on **real-world deployability** rather than benchmark-only performance.

By enabling **live, on-device deepfake detection**, it addresses a growing need for scalable AI safety tools in modern digital platforms.

---

<div align="center">

**Built with ❤️ for the Gemini 3 Hackathon**

[⬆ Back to Top](#️-deepguard)

</div>
