# Padel Game Analytics — Shot Classification System

> **AI/ML Internship Assignment — Layman AI**  
> A computer vision prototype that analyses padel match footage, detects players and the ball, classifies shot types, and exports structured results.

---

## Table of contents

1. [Project overview](#project-overview)
2. [System architecture](#system-architecture)
3. [Features](#features)
4. [Tech stack](#tech-stack)
5. [Setup & installation](#setup--installation)
6. [Usage](#usage)
7. [Output format](#output-format)
8. [Methodology](#methodology)
9. [Challenges faced](#challenges-faced)
10. [Improvements I would make](#improvements-i-would-make)
11. [Project structure](#project-structure)
12. [Model links](#model-links)

---

## Project overview

This system takes a padel match video as input and produces:

- Frame-by-frame detection of the **ball**, **rackets**, and **players**
- **Shot classification** into three types: Forehand, Backhand, Serve/Smash
- **Annotated output video** with bounding boxes and shot labels overlaid
- **Structured JSON and CSV files** with shot events (type, timestamp, player)
- **Analytics dashboard** with shot count charts and player statistics

---

## System architecture

```
┌─────────────────────────────────────────────────────┐
│                    INPUT LAYER                      │
│            Padel video (mp4) + config.py            │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│                  CV PIPELINE LAYER                  │
│  detection.py   │  tracking.py  │ pose_estimator.py │
│  YOLOv8         │  ByteTrack    │ MediaPipe Pose     │
│  (ball/players) │  (stable IDs) │ (body keypoints)  │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│              CLASSIFICATION LAYER                   │
│                 shot_classifier.py                  │
│   Rule-based: wrist angle + velocity → shot type    │
│   Forehand │ Backhand │ Serve / Smash               │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│               ANALYTICS LAYER                       │
│         analytics.py + visualizer.py               │
│   Shot counts │ Rally stats │ Video frame overlay   │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│                 EXPORT LAYER                        │
│                   exporter.py                       │
│   shots.json │ shots.csv │ output_video.mp4         │
└─────────────────────────────────────────────────────┘
```

**Data flow per frame:**

```
Frame → YOLOv8 detect → ByteTrack assign IDs → MediaPipe extract keypoints
     → Rule classifier → Shot event emitted → Overlay drawn → Export written
```

---

## Features

### Core (mandatory)

- [x] Ball, racket, and player detection using YOLOv8
- [x] Multi-object tracking with consistent player IDs (ByteTrack)
- [x] Pose estimation using MediaPipe (wrist, elbow, shoulder keypoints)
- [x] Shot classification — Forehand, Backhand, Serve/Smash
- [x] JSON output with `shot_type`, `frame`, `timestamp_sec`, `player_id`
- [x] CSV output for tabular analysis

### Bonus

- [x] Shot count analytics per player (forehand vs backhand totals)
- [x] Annotated output video with bounding boxes and labels
- [x] Shot timeline chart (matplotlib)
- [x] Player court positioning heatmap (seaborn)
- [x] Rule-based bounce/direction detection

---

## Tech stack

| Component | Library / Tool |
|-----------|---------------|
| Language | Python 3.10+ |
| Object detection | YOLOv8 (Ultralytics) |
| Object tracking | ByteTrack (built into Ultralytics) |
| Pose estimation | MediaPipe |
| Video I/O | OpenCV |
| Data processing | NumPy, Pandas, SciPy |
| Visualisation | Matplotlib, Seaborn |
| Notebooks | JupyterLab |
| Testing | pytest |

---

## Setup & installation

### Prerequisites

- Python 3.10 or higher
- pip
- Git
- A GPU is recommended for faster inference but the system runs on CPU too

### 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/padel-analytics.git
cd padel-analytics
```

### 2 — Create a virtual environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS / Linux
source venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Download pretrained models

The YOLOv8 nano model (`yolov8n.pt`) is downloaded automatically on first run.  
Custom fine-tuned models (if available) — see [Model links](#model-links).

### 5 — Add your video

Place your padel match video in `data/raw/`. The default expected filename is `match.mp4`.  
Update the path in `src/config.py` if needed.

---

## Usage

### Run the full pipeline

```bash
python main.py --input data/raw/match.mp4 --output data/outputs/
```

### Optional flags

```bash
python main.py \
  --input  data/raw/match.mp4 \   # path to input video
  --output data/outputs/ \         # directory for all outputs
  --save-video \                   # write annotated video (default: True)
  --conf   0.4 \                   # YOLOv8 confidence threshold
  --device cpu                     # 'cpu' or '0' for first GPU
```

### Run notebooks

```bash
jupyter lab
```

Then open:

- `notebooks/01_exploration.ipynb` — frame-level EDA
- `notebooks/02_model_training.ipynb` — custom classifier training (optional)
- `notebooks/03_analytics_dashboard.ipynb` — shot stats and charts

### Run tests

```bash
pytest tests/ -v
```

---

## Output format

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
  },
  {
    "frame": 287,
    "timestamp_sec": 11.96,
    "player_id": 2,
    "shot_type": "backhand",
    "confidence": 0.76,
    "wrist_velocity": 14.7,
    "wrist_angle_deg": -28.5
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

## Methodology

### Step 1 — Object detection

YOLOv8 (`yolov8n.pt`) runs on every frame to detect `person` class bounding boxes. A sports-specific checkpoint can optionally detect the padel ball directly; otherwise, ball detection falls back to a small circular blob detector using OpenCV contours.

### Step 2 — Multi-object tracking

ByteTrack (integrated in Ultralytics) assigns stable integer IDs to each detected player across frames, handling occlusions and re-entries gracefully. This allows per-player shot attribution.

### Step 3 — Pose estimation

For each tracked player crop, MediaPipe Pose extracts 33 body landmarks. We focus on three keypoints: wrist, elbow, and shoulder of the dominant arm.

### Step 4 — Shot classification

A rule-based classifier converts keypoint sequences into shot labels:

| Shot type | Rule |
|-----------|------|
| **Forehand** | Wrist moves left-to-right (relative to player facing direction), angle > 0° |
| **Backhand** | Wrist moves right-to-left, angle < 0° |
| **Serve / Smash** | Wrist rises above shoulder level at high velocity (> threshold) |

Velocity is computed as the Euclidean distance between wrist positions in consecutive frames, smoothed with a rolling mean over 5 frames to reduce noise.

A debounce window of 15 frames prevents the same shot from being counted multiple times.

### Step 5 — Analytics & export

Shot events are aggregated into summary statistics per player. The visualizer draws bounding boxes, player IDs, and the most recent shot label onto each frame. Results are written to JSON, CSV, and an annotated MP4.

---

## Challenges faced

**1. Ball detection reliability**  
The padel ball is very small (8–10 pixels at typical broadcast resolution) and moves extremely fast, causing motion blur. YOLOv8 nano often misses it. Workaround: used OpenCV contour-based circular blob detection as a fallback, accepting some false positives.

**2. Shot boundary determination**  
Distinguishing when one shot ends and the next begins was harder than expected. The same wrist motion can persist for 20–30 frames. Solved with a debounce window and minimum inter-event distance.

**3. Player orientation**  
The forehand/backhand distinction depends on which side of the body the shot is hit from, which itself depends on which direction the player is facing. The current rule-based approach assumes a fixed camera angle. A more robust solution would estimate player heading from hip and shoulder vectors.

**4. Absence of a labelled padel dataset**  
No public annotated padel video dataset exists. Shot labels were assigned manually to a short validation clip for evaluation. This limits quantitative accuracy assessment.

---

## Improvements I would make

1. **Train a dedicated ball detection model** — fine-tune YOLOv8 on a padel-specific dataset with small-ball annotations, or use TrackNet (designed for sports ball tracking).

2. **Replace rule-based classifier with an LSTM / TCN** — treat keypoint sequences as time series and train a sequence model on labelled examples. This would generalise better across player styles and camera angles.

3. **Player heading estimation** — use hip-to-shoulder vector to determine which side of the body a shot is struck from, removing the fixed-camera-angle assumption.

4. **Multi-camera support** — padel courts often have multiple cameras. Fusing views would improve ball tracking during fast rallies.

5. **Real-time inference mode** — optimise the pipeline with model quantisation (INT8) and frame skipping to support live-stream analysis.

6. **Web dashboard** — replace the static Jupyter dashboard with a Streamlit or FastAPI app for interactive match review.

---

## Project structure

```
padel-analytics/
├── data/
│   ├── raw/             # original input videos
│   ├── processed/       # extracted frames
│   ├── outputs/         # shots.json, shots.csv, output video
│   └── annotations/     # manual labels for evaluation
├── models/
│   ├── detection/       # YOLO weights (.pt files)
│   ├── classification/  # custom classifier weights (if trained)
│   └── pretrained/      # downloaded base models
├── src/
│   ├── config.py        # all paths, thresholds, constants
│   ├── detection.py     # YOLOv8 inference wrapper
│   ├── tracking.py      # ByteTrack integration
│   ├── pose_estimator.py# MediaPipe pose extraction
│   ├── shot_classifier.py# rule-based + ML shot classification
│   ├── analytics.py     # shot counts, rally stats, heatmaps
│   ├── visualizer.py    # frame overlay drawing
│   ├── exporter.py      # JSON / CSV / video writer
│   └── utils.py         # shared helpers (logger, geometry utils)
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_analytics_dashboard.ipynb
├── tests/
│   ├── test_detection.py
│   └── test_classifier.py
├── main.py              # CLI entry point
├── requirements.txt
└── README.md
```

---

## Model links

| Model | Description | Link |
|-------|-------------|------|
| `yolov8n.pt` | YOLOv8 nano — base detection | Auto-downloaded by Ultralytics |
| `padel_classifier.pt` | Custom shot classifier (if trained) | [Google Drive →](#) |

> Replace `#` above with your actual Google Drive share link before submission.

---

## Author

Built for the **Layman AI — AI/ML Internship Assignment**  
Submission deadline: May 10, 2026 (late: May 12, 2026)