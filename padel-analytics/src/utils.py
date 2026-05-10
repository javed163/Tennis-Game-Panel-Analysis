"""
utils.py
========
Shared utility functions used across the entire pipeline:
  - Logger setup
  - Geometry helpers  (distance, angle, midpoint)
  - Frame helpers     (resize, timestamp formatting)
  - Validation        (check file exists, video readable)
"""

import math
import logging
import sys
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np

from config import LOG_LEVEL, LOG_FILE


# ─── Logger ───────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger that writes to both stdout and a log file.
    Call once per module:  logger = get_logger(__name__)
    """
    logger = logging.getLogger(name)

    if logger.handlers:          # avoid adding duplicate handlers
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    try:
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError:
        pass   # if log file can't be created, just use console

    return logger


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def euclidean_distance(p1: Tuple[float, float],
                       p2: Tuple[float, float]) -> float:
    """Straight-line distance between two (x, y) points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def angle_degrees(p1: Tuple[float, float],
                  p2: Tuple[float, float]) -> float:
    """
    Angle (degrees) of the vector from p1 → p2 relative to vertical.
    Positive = right of vertical, Negative = left of vertical.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    # atan2 measured from positive x-axis; convert to angle from vertical
    angle = math.degrees(math.atan2(dx, -dy))
    return angle


def midpoint(p1: Tuple[float, float],
             p2: Tuple[float, float]) -> Tuple[float, float]:
    """Return the midpoint between two (x, y) points."""
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Return the center pixel of a bounding box.
    bbox format: (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    """Return pixel area of a bounding box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value into [lo, hi]."""
    return max(lo, min(hi, value))


# ─── Smoothing ────────────────────────────────────────────────────────────────

def rolling_mean(values: list, window: int) -> list:
    """
    Apply a simple rolling (box) mean to a list of floats.
    Returns a list of the same length; edges use available values.
    """
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start: i + 1]
        result.append(sum(chunk) / len(chunk))
    return result


# ─── Frame helpers ────────────────────────────────────────────────────────────

def resize_frame(frame: np.ndarray,
                 width: Optional[int] = None,
                 height: Optional[int] = None) -> np.ndarray:
    """
    Resize a frame while preserving aspect ratio.
    Supply either width OR height (not both).
    """
    h, w = frame.shape[:2]
    if width is None and height is None:
        return frame
    if width is not None:
        ratio = width / w
        new_size = (width, int(h * ratio))
    else:
        ratio = height / h
        new_size = (int(w * ratio), height)
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)


def frame_to_timestamp(frame_idx: int, fps: float) -> str:
    """Convert a frame index to a human-readable MM:SS.mmm string."""
    total_seconds = frame_idx / fps if fps > 0 else 0
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:06.3f}"


def timestamp_seconds(frame_idx: int, fps: float) -> float:
    """Convert a frame index to decimal seconds."""
    return round(frame_idx / fps, 4) if fps > 0 else 0.0


# ─── Validation ───────────────────────────────────────────────────────────────

def validate_video(path: str | Path) -> Tuple[bool, str]:
    """
    Check that a video file exists and can be opened by OpenCV.
    Returns (ok: bool, message: str).
    """
    path = Path(path)
    if not path.exists():
        return False, f"File not found: {path}"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False, f"OpenCV cannot open: {path}"
    ret, _ = cap.read()
    cap.release()
    if not ret:
        return False, f"Cannot read frames from: {path}"
    return True, "OK"


def get_video_info(path: str | Path) -> dict:
    """
    Return basic metadata about a video file.
    Keys: width, height, fps, frame_count, duration_sec
    """
    cap = cv2.VideoCapture(str(path))
    info = {
        "width":        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":          cap.get(cv2.CAP_PROP_FPS),
        "frame_count":  int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_sec": 0.0,
    }
    if info["fps"] > 0:
        info["duration_sec"] = round(info["frame_count"] / info["fps"], 2)
    cap.release()
    return info


# ─── Normalisation helpers ────────────────────────────────────────────────────

def normalize_landmark(landmark, frame_w: int, frame_h: int
                        ) -> Tuple[float, float]:
    """
    Convert a MediaPipe NormalizedLandmark (0–1 range) to pixel coords.
    Returns (x_px, y_px).
    """
    return (landmark.x * frame_w, landmark.y * frame_h)