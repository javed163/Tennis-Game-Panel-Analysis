"""
config.py
=========
Single source of truth for all project paths, model settings,
detection thresholds, and classification parameters.
Edit values here — never hardcode them elsewhere.
"""

import os
from pathlib import Path

# ─── Project root ─────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent   # padel-analytics/

# ─── Data directories ─────────────────────────────────────────────────────────
DATA_DIR        = ROOT_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
OUTPUTS_DIR     = DATA_DIR / "outputs"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

# ─── Model directories ────────────────────────────────────────────────────────
MODELS_DIR          = ROOT_DIR / "models"
DETECTION_DIR       = MODELS_DIR / "detection"
CLASSIFICATION_DIR  = MODELS_DIR / "classification"
PRETRAINED_DIR      = MODELS_DIR / "pretrained"

# ─── Auto-create all directories on import ────────────────────────────────────
for _dir in [
    RAW_DIR, PROCESSED_DIR, OUTPUTS_DIR, ANNOTATIONS_DIR,
    DETECTION_DIR, CLASSIFICATION_DIR, PRETRAINED_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─── Input / Output files ─────────────────────────────────────────────────────
# Default video path — override with --input flag on the CLI.
# Supports Tennis folder: python main.py --input Tennis/your_video.mp4
DEFAULT_INPUT_VIDEO  = RAW_DIR / "jetbrains://pycharm/navigate/reference?project=Panel%20Analytics&path=padel-analytics%2Fdata%2Fraw%2Finfernce_sample_video.mp4"

# ── Tennis video folder support ───────────────────────────────────────────────
# If you have videos in a Tennis/ folder (e.g. Tennis/split_data/),
# pass the path directly:
#   python main.py --input Tennis/split_data/video_001.mp4
# The pipeline accepts any .mp4 / .avi / .mov file anywhere on disk.
TENNIS_DIR = ROOT_DIR / "Tennis"
OUTPUT_VIDEO         = OUTPUTS_DIR / "output_annotated.mp4"
OUTPUT_JSON          = OUTPUTS_DIR / "shots.json"
OUTPUT_CSV           = OUTPUTS_DIR / "shots.csv"

# ─── YOLO Detection settings ──────────────────────────────────────────────────
YOLO_MODEL_NAME      = "yolov8n.pt"          # auto-downloaded by Ultralytics
YOLO_MODEL_PATH      = PRETRAINED_DIR / YOLO_MODEL_NAME
YOLO_CONF_THRESHOLD  = 0.40                   # minimum detection confidence
YOLO_IOU_THRESHOLD   = 0.45                   # NMS IoU threshold
YOLO_CLASSES         = [0]                    # 0 = person (COCO class)
YOLO_IMG_SIZE        = 640                    # inference image size (px)
YOLO_DEVICE          = "cpu"                  # "cpu" | "0" (first GPU) | "cuda"

# ─── Tracking settings ────────────────────────────────────────────────────────
TRACKER_TYPE         = "bytetrack"            # built into Ultralytics
MAX_PLAYERS          = 4                      # max simultaneous tracked players

# ─── MediaPipe Pose settings ──────────────────────────────────────────────────
POSE_MIN_DETECTION_CONF  = 0.5
POSE_MIN_TRACKING_CONF   = 0.5
POSE_MODEL_COMPLEXITY    = 1                  # 0 = fast, 1 = balanced, 2 = accurate

# ─── Shot classification thresholds ───────────────────────────────────────────
# Wrist velocity (pixels / frame, after smoothing) to trigger a shot event
SHOT_VELOCITY_THRESHOLD  = 8.0

# Wrist-angle (degrees, relative to vertical) boundaries
FOREHAND_ANGLE_MIN   =  10.0    # wrist angle > +10° → forehand
BACKHAND_ANGLE_MAX   = -10.0    # wrist angle < -10° → backhand
SMASH_HEIGHT_RATIO   =  0.85    # wrist y < shoulder_y * this ratio → smash

# Debounce: ignore repeated shots within this many frames
SHOT_DEBOUNCE_FRAMES = 15

# Smoothing window for velocity calculation (frames)
VELOCITY_SMOOTH_WINDOW = 5

# ─── Visualizer settings ──────────────────────────────────────────────────────
BBOX_THICKNESS       = 2
FONT_SCALE           = 0.6
FONT_THICKNESS       = 2

# Colour palette (BGR for OpenCV)
COLOURS = {
    "player":    (0,   200, 100),   # green
    "ball":      (0,   165, 255),   # orange
    "forehand":  (255, 100,   0),   # blue
    "backhand":  (0,   100, 255),   # red-orange
    "smash":     (0,   0,   255),   # red
    "default":   (200, 200, 200),   # grey
}

# ─── Video writer settings ────────────────────────────────────────────────────
OUTPUT_FPS       = 30              # frames per second of output video
OUTPUT_FOURCC    = "mp4v"          # codec — mp4v for .mp4

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL        = "INFO"          # DEBUG | INFO | WARNING | ERROR
LOG_FILE         = OUTPUTS_DIR / "pipeline.log"