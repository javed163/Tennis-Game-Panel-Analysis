
# 🎾 Padel Analytics — AI-Powered Shot Classification System

> **Real-time padel match analysis** using computer vision and machine learning  
> Detect players • Track movement • Classify shots • Generate insights

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Detection-green)](https://github.com/ultralytics/ultralytics)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose%20Estimation-orange)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#license)

---
This is the link of sample output video 
https://drive.google.com/file/d/18XyHSTeX5cxJGMyDaqG8B8W49LEpEmlI/view?usp=sharing

## 📋 Overview

**Padel Analytics** is an AI-powered system that analyzes padel match videos and provides frame-by-frame insights. Feed it a video, and get:

- **🎯 Player & Ball Detection** — Locates all players and the ball on court
- **🔄 Multi-Player Tracking** — Assigns unique IDs to each player across frames
- **💪 Pose Estimation** — Extracts body keypoints (wrist, elbow, shoulder)
- **⚡ Shot Classification** — Automatically labels shots as Forehand, Backhand, or Smash
- **📊 Analytics Dashboard** — Shot counts, rally statistics, player heatmaps
- **📹 Annotated Video Export** — Output video with bounding boxes and shot labels
- **📄 Structured Data** — JSON & CSV exports for further analysis

### Example Output

| Shot Event | Timestamp | Player | Type |
|-----------|-----------|--------|------|
| Frame 142 | 5.92s | Player 1 | Forehand |
| Frame 287 | 11.96s | Player 2 | Backhand |
| Frame 512 | 21.33s | Player 1 | Smash |

---

## 🏗️ System Architecture

```
INPUT VIDEO
    ↓
[DETECTION] YOLOv8 → Detect players, ball, rackets
    ↓
[TRACKING] ByteTrack → Assign stable player IDs across frames
    ↓
[POSE ESTIMATION] MediaPipe → Extract wrist, elbow, shoulder keypoints
    ↓
[CLASSIFICATION] Rule-based Engine → Forehand? Backhand? Smash?
    ↓
[ANALYTICS] Shot aggregation, rally stats, heatmaps
    ↓
[VISUALIZATION] Draw overlays, bounding boxes, labels
    ↓
[EXPORT] JSON, CSV, annotated MP4, charts
```

---

## ✨ Key Features

### Core Capabilities
- ✅ YOLOv8-based object detection (players, ball, rackets)
- ✅ Multi-object tracking with persistent player IDs
- ✅ MediaPipe pose estimation (33 body landmarks)
- ✅ Rule-based shot classification (Forehand/Backhand/Smash)
- ✅ JSON & CSV export with shot metadata
- ✅ Annotated output video with overlays

### Advanced Analytics
- 📊 Shot count breakdown per player
- 📈 Rally duration and frequency analysis
- 🔥 Player court positioning heatmap
- 📉 Shot timeline visualization
- 🎬 Frame-by-frame shot confidence scores

---

## 🚀 Quick Start

### Prerequisites
```bash
• Python 3.10+
• Git
• GPU recommended (but CPU works fine)
```

### 1️⃣ Clone & Setup
```bash
git clone https://github.com/yourusername/padel-analytics.git
cd padel-analytics
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Add Your Video
```bash
# Place video in data/raw/ folder
cp your_match.mp4 data/raw/match.mp4
```

### 4️⃣ Run Analysis
```bash
# Basic run
python main.py --input data/raw/match.mp4

# Full options
python main.py \
  --input data/raw/match.mp4 \
  --output data/outputs/ \
  --conf 0.4 \
  --device cpu
```

### 5️⃣ View Results
Results appear in `data/outputs/`:
- `output_annotated.mp4` — Video with overlays
- `shots.json` — Structured shot events
- `shots.csv` — Tabular shot data
- `analytics.html` — Interactive dashboard

---

## 📊 Output Format

### shots.json
```json
[
  {
    "frame": 142,
    "timestamp_sec": 5.92,
    "player_id": 1,
    "shot_type": "forehand",
    "confidence": 0.84,
    "wrist_velocity": 18.3,
    "wrist_angle_deg": 32.1
  }
]
```

### shots.csv
```
frame,timestamp_sec,player_id,shot_type,confidence,wrist_velocity,wrist_angle_deg
142,5.92,1,forehand,0.84,18.3,32.1
287,11.96,2,backhand,0.76,14.7,-28.5
```

---

## 📹 Adding Video Demo / Links to README

### Option 1: Embed YouTube Video
```markdown
### Demo Video
[![Watch Demo](https://www.youtube.com/watch?v=L23oIHZE14w&t=13272s)

*Take a reference from this youtube channel*
```

---

## 🛠️ Configuration

Edit `src/config.py` to customize:
```python
# Detection thresholds
YOLO_CONF_THRESHOLD = 0.4        # Increase for fewer false positives
YOLO_DEVICE = "cpu"              # "cpu" or "0" for GPU

# Classification
VELOCITY_THRESHOLD = 8.0         # Wrist velocity trigger
DEBOUNCE_FRAMES = 15             # Frames between same-player shots

# Output
OUTPUT_FPS = 30                   # Output video framerate
```

---

## 🔬 Methodology

### Shot Detection Pipeline

1. **Object Detection** → YOLOv8 scans each frame
2. **Multi-Tracking** → ByteTrack assigns persistent player IDs
3. **Pose Extraction** → MediaPipe gets wrist/elbow/shoulder positions
4. **Classification** → Rule-based engine computes shot type:
   - **Forehand**: Wrist moves left-to-right, angle > 0°
   - **Backhand**: Wrist moves right-to-left, angle < 0°
   - **Smash**: Wrist rises above shoulder at high velocity

5. **Analytics** → Aggregate shots by player and type

### Velocity Calculation
```
velocity = euclidean_distance(wrist_frame_n, wrist_frame_n-1)
smoothed_velocity = rolling_mean(velocity, window=5)
```

---

## 📁 Project Structure

```
padel-analytics/
├── data/
│   ├── raw/              # Input videos
│   ├── processed/        # Extracted frames
│   ├── outputs/          # Results (JSON, CSV, video)
│   └── annotations/      # Manual labels
├── src/
│   ├── config.py         # Configuration & paths
│   ├── detection.py      # YOLOv8 wrapper
│   ├── tracking.py       # ByteTrack integration
│   ├── pose_estimator.py # MediaPipe interface
│   ├── shot_classifier.py# Shot classification logic
│   ├── analytics.py      # Statistics & aggregation
│   ├── visualizer.py     # Frame overlay drawing
│   └── exporter.py       # JSON/CSV/video export
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_analytics_dashboard.ipynb
├── tests/
│   ├── test_detection.py
│   └── test_classifier.py
├── main.py               # CLI entry point
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

---

## 🧠 How It Works

### Per-Frame Processing
```
Frame → YOLOv8 detect 
     → ByteTrack assign IDs 
     → MediaPipe extract keypoints
     → Rule classifier → Shot event
     → Draw overlays
     → Write to output
```

### Shot Classification Rules

| Shot Type | Conditions |
|-----------|-----------|
| **Forehand** | Wrist accelerates left-to-right; angle > 0° |
| **Backhand** | Wrist accelerates right-to-left; angle < 0° |
| **Smash** | Wrist velocity > threshold; shoulder height |

Shots are debounced (15-frame window) to avoid duplicate detections.

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_classifier.py -v

# Run with coverage
pytest tests/ --cov=src
```

---

## 📦 Tech Stack

| Component | Tool |
|-----------|------|
| **Language** | Python 3.10+ |
| **Detection** | YOLOv8 (Ultralytics) |
| **Tracking** | ByteTrack |
| **Pose Est.** | MediaPipe |
| **Video I/O** | OpenCV |
| **Data** | NumPy, Pandas, SciPy |
| **Visualization** | Matplotlib, Seaborn |
| **Testing** | pytest |

---

## 🤔 FAQ

**Q: Can it work on CPU?**  
A: Yes! GPU is recommended for speed (~30 fps), but CPU works (~1-3 fps).

**Q: What video formats are supported?**  
A: `.mp4`, `.avi`, `.mov` — any OpenCV-compatible format.

**Q: How do I improve accuracy?**  
A: See [Improvements](#improvements) section below.

**Q: Can it run on live streams?**  
A: Not yet, but batch processing is fully supported.

---

## 🎯 Future Improvements

1. **Train Custom Ball Detector** — Fine-tune YOLOv8 on padel-specific dataset
2. **Deep Learning Classifier** — Replace rule-based engine with LSTM/TCN
3. **Player Heading Estimation** — Auto-detect player orientation
4. **Multi-Camera Fusion** — Combine views from multiple angles
5. **Real-Time Mode** — Optimized inference with quantization
6. **Web Dashboard** — Interactive Streamlit/FastAPI interface

---

## ⚠️ Known Limitations

- **Small ball** — Padel ball (8-10px) can be missed in motion blur
- **Fixed camera angle** — Rule-based classifier assumes static viewpoint
- **No annotated dataset** — Limits supervised learning approaches
- **Single camera** — Doesn't handle multi-view scenarios

---

## 📝 Citation

If you use this project in research, please cite:

```bibtex
@software{padel_analytics_2026,
  author = {Javed Ahamad Husen},
  title = {Padel Analytics - AI-Powered Shot Classification System},
  year = {2026},
  url = {https://github.com/javed163/Tennis-Game-Panel-Analysis}
}
```

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) file for details.

---

## 👤 Author

Built as an **AI/ML Internship Assignment** for **Layman AI**  

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📞 Support

- 📧 Email: javedahamad906070@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/javed163/Tennis-Game-Panel-Analysis/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/javed163/Tennis-Game-Panel-Analysis/discussions)

---

## ⭐ Star History

If you find this useful, please consider starring the repo! ⭐

---

**Made with ❤️ for padel enthusiasts and AI/ML engineers**
