# 🛡️ DeepGuard v2 - Real-Time Deepfake Detection Shield

<div align="center">

![DeepGuard Banner](https://img.shields.io/badge/DeepGuard-v2.0-blue?style=for-the-badge&logo=shield)
[![Gemini 3 Hackathon](https://img.shields.io/badge/Gemini%203-Hackathon-4285F4?style=for-the-badge&logo=google)](https://gemini3.devpost.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![PyQt6](https://img.shields.io/badge/PyQt6-6.5+-41CD52?style=for-the-badge&logo=qt&logoColor=white)](https://riverbankcomputing.com/software/pyqt/)

**A real-time AI safety layer that detects deepfakes as you scroll.**

*Built for the [Gemini 3 Hackathon](https://gemini3.devpost.com/) by Google DeepMind*

[🎬 Demo Video](#-demo) • [🚀 Quick Start](#-quick-start) • [🏗️ Architecture](#-architecture) • [🤖 Gemini Integration](#-gemini-integration)

</div>

---

## 🎯 The Problem

We live in the **"scroll era"** - endless feeds of Reels, Shorts, and TikToks. Deepfakes blend seamlessly into this content, and traditional detection tools require you to:
- Stop scrolling
- Copy/download the video
- Upload to a website
- Wait for analysis

**Nobody does this.** The friction is too high. By the time you verify, you've already moved on.

## 💡 The Solution

DeepGuard v2 is a **real-time deepfake detection overlay** that:
- 🛡️ **Sits on top of your screen** - Always watching, never intrusive
- ⚡ **Analyzes in real-time** - No uploading, no waiting
- 🎨 **Shows confidence levels** - 5-tier system from 🟢 REAL to 🔴 DEEPFAKE
- 🤖 **Explains with Gemini** - Human-readable explanations powered by Google's AI
- 📱 **Works everywhere** - Instagram, YouTube, TikTok, any website, any app

**Think of it as an antivirus for deepfakes.**

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DeepGuard v2 Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Screen     │───▶│    Face      │───▶│   MesoNet    │       │
│  │   Capture    │    │  Detection   │    │    Model     │       │
│  │   (mss)      │    │   (MTCNN)    │    │  (256x256)   │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────────────────────────────────────────────┐       │
│  │              Temporal Aggregation Engine              │       │
│  │  • 30-frame sliding window                            │       │
│  │  • Weighted averaging (exponential decay)             │       │
│  │  • Trend detection (rising/falling/stable)            │       │
│  └──────────────────────────────────────────────────────┘       │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  Confidence  │───▶│   Gemini     │───▶│   Overlay    │       │
│  │  Classifier  │    │  Explainer   │    │   Window     │       │
│  │  (5-tier)    │    │  (API)       │    │   (PyQt6)    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Screen Capture** | `mss` | Captures screen at 10+ FPS with thread-safety |
| **Face Detection** | MTCNN | Locates faces in captured frames |
| **Deepfake Model** | MesoNet (Keras) | CNN trained on FaceForensics++ dataset |
| **Temporal Engine** | Custom | Smooths predictions over 30 frames |
| **Confidence System** | Custom | Maps scores to 5 human-readable levels |
| **Explainer** | **Gemini API** | Generates natural language explanations |
| **Overlay** | PyQt6 | Floating, draggable, always-on-top window |

---

## 🤖 Gemini Integration

**Gemini is central to DeepGuard's user experience.**

### How Gemini is Used

DeepGuard's ML pipeline produces technical outputs (probability scores, frame counts, trends). Gemini transforms these into **human-understandable explanations**:

```python
# Technical Detection Result
{
    "level": "LIKELY_FAKE",
    "score": 0.72,
    "confidence_pct": 78,
    "trend": "rising",
    "frames_analyzed": 30
}

# Gemini Transforms To:
"This video shows signs of digital manipulation. The face movements 
appear inconsistent with natural expressions. Consider verifying 
this content from trusted sources before sharing."
```

### Why Gemini?

1. **Real-time speed** - `gemini-2.0-flash-exp` provides sub-second responses
2. **Contextual understanding** - Adapts explanations based on confidence levels and trends
3. **User-friendly** - No technical jargon, actionable advice
4. **Graceful degradation** - Falls back to deterministic templates if API unavailable

### Gemini API Code

```python
# From core/explainer.py
from google import genai

class GeminiExplainer:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    def explain(self, result: DetectionResult) -> str:
        prompt = f"""You are an AI safety assistant explaining deepfake detection.
        
        Classification: {result.level.value}
        Confidence: {result.confidence_pct}%
        Trend: {result.trend}
        
        Generate a brief, helpful explanation (2-3 sentences) for a non-technical user."""
        
        response = self.client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt
        )
        return response.text
```

### Gemini Features Used

- ✅ **Text Generation** - Natural language explanations
- ✅ **Fast Inference** - Real-time responses with Flash model
- ✅ **Safety Filtering** - Built-in content safety
- ✅ **Structured Prompting** - Consistent, contextual outputs

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Windows 10/11 (overlay optimized for Windows)
- Gemini API Key ([Get one free](https://aistudio.google.com/apikey))

### Installation

```bash
# Clone the repository
git clone https://github.com/omtripathi52/deepguard.git
cd deepguard

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set Gemini API key
set GEMINI_API_KEY=your_api_key_here  # Windows
# export GEMINI_API_KEY=your_api_key_here  # Linux/Mac

# Run DeepGuard v2 Overlay
python main.py
```

---

## 🎯 Usage Modes

### 🆕 Mode 1: Real-Time Overlay (v2)

```bash
python main.py
```

- Floating overlay appears in top-right corner
- Browse Instagram, YouTube, TikTok normally
- Watch the shield change colors based on detection
- Hover for Gemini explanation
- Drag to reposition
- Click ✕ to close

**Confidence Levels:**
- 🟢 **REAL** - Authentic content
- 🟢 **LIKELY REAL** - Probably authentic  
- 🟡 **UNCERTAIN** - Cannot determine
- 🟠 **LIKELY FAKE** - Suspicious content
- 🔴 **DEEPFAKE** - Likely manipulated

### 🖼️ Mode 2: Image Detection

```bash
python -m core.image_pipeline --image path/to/image.jpg
```

### 🎞️ Mode 3: Video Detection

```bash
python -m core.video_pipeline --video path/to/video.mp4
```

### 📷 Mode 4: Webcam Detection

```bash
python -m core.live_pipeline
```
Press `q` to quit

### 🖥️ Mode 5: Screen Pipeline (CLI)

```bash
python -m core.screen_pipeline
```
Press `Ctrl+C` to stop

---

## 📁 Project Structure

```
deepguard/
├── main.py                 # 🆕 v2 Entry point - overlay mode
├── config.py               # 🆕 Centralized configuration
├── requirements.txt        # Dependencies
├── README.md               # This file
│
├── core/                   # Detection engine
│   ├── engine.py          # 🆕 v2 Main orchestration
│   ├── confidence.py      # 🆕 5-tier classification
│   ├── temporal.py        # 🆕 Frame aggregation
│   ├── screen_capture.py  # 🆕 Thread-safe screen capture
│   ├── explainer.py       # 🆕 Gemini integration
│   ├── detector.py        # Core DeepfakeDetector
│   ├── face_detector_mtcnn.py  # MTCNN wrapper
│   ├── image_pipeline.py  # Image analysis
│   ├── video_pipeline.py  # Video processing
│   ├── live_pipeline.py   # Webcam detection
│   └── screen_pipeline.py # CLI screen capture
│
├── overlay/                # 🆕 PyQt6 UI
│   ├── __init__.py
│   └── window.py          # Floating overlay
│
└── mesonet/               # MesoNet model
    ├── classifiers.py     # Model architecture
    └── weights/           # Pre-trained weights
```

---

## 🔧 Configuration

All settings in `config.py`:

```python
# Capture settings
capture.fps = 10                    # Analysis framerate
capture.monitor_index = 1           # Which monitor to capture

# Detection thresholds (probability of FAKE)
confidence.real_high = 0.20         # Below = REAL
confidence.fake_low = 0.65          # Above = DEEPFAKE

# Overlay appearance
overlay.position = "top-right"      # Window position
overlay.opacity = 0.92              # Transparency

# Gemini API
gemini.model = "gemini-2.0-flash-exp"  # Latest experimental model
gemini.enabled = True
```

---

## 🎬 Demo

[📺 Watch the 3-minute demo video](https://youtube.com/your-demo-link)

*Shows DeepGuard detecting deepfakes across Instagram Reels, YouTube Shorts, and TikTok*

---

## 📊 Judging Criteria Alignment

| Criteria | Weight | How DeepGuard Addresses It |
|----------|--------|---------------------------|
| **Technical Execution** | 40% | Working end-to-end pipeline with real-time ML inference, temporal smoothing, and robust error handling |
| **Innovation/Wow Factor** | 30% | Novel "always-on overlay" approach vs traditional upload-analyze pattern |
| **Potential Impact** | 20% | Protects against misinformation spread across all social platforms |
| **Presentation/Demo** | 10% | Clean UI, comprehensive documentation, clear demo video |

---

## 🔮 Future Roadmap

- [ ] **Android App** - Mobile deepfake detection
- [ ] **Browser Extension** - Chrome/Firefox integration
- [ ] **API Service** - Detection as a service
- [ ] **Multi-model Ensemble** - Combine multiple detectors
- [ ] **GPU Acceleration** - Faster inference with CUDA

---

## 📜 Third-Party Libraries

| Library | License | Purpose |
|---------|---------|---------|
| TensorFlow | Apache 2.0 | Deep learning framework |
| PyQt6 | GPL v3 | UI framework |
| MTCNN | MIT | Face detection |
| mss | MIT | Screen capture |
| google-genai | Apache 2.0 | Gemini API client |

---

## 👨‍💻 Author

Built with ❤️ for the Gemini 3 Hackathon

---

## 📄 License

Apache 2.0 - See [LICENSE](LICENSE) for details.

---

<div align="center">

**🛡️ Stay safe. Stay informed. Stay protected.**

*DeepGuard v2 - Your real-time shield against deepfakes*

</div>
