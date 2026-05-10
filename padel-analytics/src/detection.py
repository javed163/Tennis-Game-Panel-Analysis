"""
detection.py
============
YOLOv8-based object detector for padel video frames.

Detects:
  - Players  (COCO class 0  — person)
  - Sports ball (COCO class 32 — sports ball, fallback to blob detector)
  - Rackets  (approximated from player crop when no dedicated class exists)

Public API
----------
    detector = Detector()
    results  = detector.detect(frame)          # returns DetectionResult
    detector.close()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Ultralytics import (yolov8)
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

from config import (
    YOLO_MODEL_NAME,
    YOLO_MODEL_PATH,
    YOLO_CONF_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    YOLO_IMG_SIZE,
    YOLO_DEVICE,
    YOLO_CLASSES,
)
from utils import get_logger, bbox_center

logger = get_logger(__name__)


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """Single detected object in one frame."""
    bbox:       Tuple[int, int, int, int]   # (x1, y1, x2, y2) in pixels
    confidence: float                        # 0.0 – 1.0
    class_id:   int                          # COCO class index
    class_name: str                          # human-readable label
    center:     Tuple[int, int] = field(init=False)

    def __post_init__(self):
        self.center = bbox_center(self.bbox)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class DetectionResult:
    """All detections found in a single frame."""
    frame_idx:  int
    players:    List[Detection] = field(default_factory=list)
    balls:      List[Detection] = field(default_factory=list)
    raw_boxes:  List[Detection] = field(default_factory=list)   # all boxes
    inference_ms: float = 0.0

    @property
    def has_players(self) -> bool:
        return len(self.players) > 0

    @property
    def has_ball(self) -> bool:
        return len(self.balls) > 0

    @property
    def player_count(self) -> int:
        return len(self.players)


# ─── Blob detector fallback (ball) ────────────────────────────────────────────

def _make_blob_detector() -> cv2.SimpleBlobDetector:
    """
    Create an OpenCV blob detector tuned for a small padel ball.
    Used as a fallback when YOLO misses the ball.
    """
    params = cv2.SimpleBlobDetector_Params()

    # Filter by colour (bright)
    params.filterByColor = True
    params.blobColor = 255

    # Filter by area (ball is small — 20 to 800 px²)
    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 800

    # Filter by circularity
    params.filterByCircularity = True
    params.minCircularity = 0.5

    # Filter by convexity
    params.filterByConvexity = True
    params.minConvexity = 0.7

    params.filterByInertia = False

    return cv2.SimpleBlobDetector_create(params)


def _detect_ball_blob(frame: np.ndarray,
                      frame_idx: int) -> List[Detection]:
    """
    Detect a bright circular ball using blob analysis on a
    grayscale + threshold version of the frame.
    Returns a list of Detection objects (may be empty).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Brighten and threshold to isolate bright round objects
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    detector = _make_blob_detector()
    keypoints = detector.detect(thresh)

    detections = []
    for kp in keypoints:
        cx, cy = int(kp.pt[0]), int(kp.pt[1])
        r = int(kp.size / 2) + 4          # add padding
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = cx + r, cy + r
        detections.append(Detection(
            bbox=(x1, y1, x2, y2),
            confidence=0.5,               # blob detector has no score
            class_id=32,
            class_name="ball",
        ))

    return detections


# ─── Main Detector class ──────────────────────────────────────────────────────

class Detector:
    """
    Wraps YOLOv8 for per-frame object detection.

    Parameters
    ----------
    model_name : str
        YOLOv8 model variant. Defaults to config.YOLO_MODEL_NAME.
        'yolov8n.pt' is downloaded automatically on first run.
    conf : float
        Confidence threshold (0–1).
    device : str
        'cpu', '0' for GPU 0, 'cuda', etc.
    use_blob_fallback : bool
        If True and no ball is found by YOLO, fall back to blob detector.
    """

    # COCO class IDs we care about
    _PERSON_ID      = 0
    _SPORTS_BALL_ID = 32

    def __init__(
        self,
        model_name:        str   = YOLO_MODEL_NAME,
        conf:              float = YOLO_CONF_THRESHOLD,
        iou:               float = YOLO_IOU_THRESHOLD,
        device:            str   = YOLO_DEVICE,
        img_size:          int   = YOLO_IMG_SIZE,
        use_blob_fallback: bool  = True,
    ):
        self.conf              = conf
        self.iou               = iou
        self.device            = device
        self.img_size          = img_size
        self.use_blob_fallback = use_blob_fallback
        self._model: Optional[object] = None

        if not _YOLO_AVAILABLE:
            logger.error(
                "ultralytics is not installed. "
                "Run: pip install ultralytics"
            )
            raise ImportError("ultralytics package required.")

        self._load_model(model_name)

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self, model_name: str) -> None:
        """Load YOLOv8 model from disk or download automatically."""
        model_path = Path(model_name)
        if not model_path.exists():
            model_path = YOLO_MODEL_PATH
        # Ultralytics auto-downloads if the file is just a name like 'yolov8n.pt'
        logger.info(f"Loading YOLO model: {model_name} on device={self.device}")
        self._model = YOLO(str(model_path) if model_path.exists() else model_name)
        logger.info("YOLO model loaded successfully.")

    # ── Core detection ────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> DetectionResult:
        """
        Run YOLOv8 inference on a single BGR frame.

        Parameters
        ----------
        frame     : np.ndarray   HxWx3 BGR image
        frame_idx : int          Frame number (for logging / output records)

        Returns
        -------
        DetectionResult with .players and .balls populated.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        t0 = time.perf_counter()

        results = self._model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.img_size,
            device=self.device,
            classes=[self._PERSON_ID, self._SPORTS_BALL_ID],
            verbose=False,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        det_result = DetectionResult(frame_idx=frame_idx,
                                     inference_ms=round(elapsed_ms, 1))

        if not results or results[0].boxes is None:
            logger.debug(f"Frame {frame_idx}: no detections.")
            return det_result

        boxes = results[0].boxes
        for box in boxes:
            coords     = box.xyxy[0].cpu().numpy().astype(int)
            conf_score = float(box.conf[0].cpu().numpy())
            cls_id     = int(box.cls[0].cpu().numpy())
            cls_name   = self._model.names.get(cls_id, str(cls_id))

            detection = Detection(
                bbox=(int(coords[0]), int(coords[1]),
                      int(coords[2]), int(coords[3])),
                confidence=round(conf_score, 3),
                class_id=cls_id,
                class_name=cls_name,
            )

            det_result.raw_boxes.append(detection)

            if cls_id == self._PERSON_ID:
                det_result.players.append(detection)
            elif cls_id == self._SPORTS_BALL_ID:
                det_result.balls.append(detection)

        # ── Blob fallback for ball ─────────────────────────────────────────
        if not det_result.has_ball and self.use_blob_fallback:
            blob_balls = _detect_ball_blob(frame, frame_idx)
            if blob_balls:
                logger.debug(
                    f"Frame {frame_idx}: YOLO missed ball, "
                    f"blob detector found {len(blob_balls)}."
                )
                det_result.balls.extend(blob_balls)

        logger.debug(
            f"Frame {frame_idx} | "
            f"players={det_result.player_count} "
            f"balls={len(det_result.balls)} "
            f"time={elapsed_ms:.1f}ms"
        )

        return det_result

    # ── Batch detection ───────────────────────────────────────────────────────

    def detect_batch(self,
                     frames: List[np.ndarray],
                     start_idx: int = 0) -> List[DetectionResult]:
        """
        Detect objects in a list of frames.
        Returns one DetectionResult per frame.
        """
        return [
            self.detect(frame, frame_idx=start_idx + i)
            for i, frame in enumerate(frames)
        ]

    # ── Convenience ───────────────────────────────────────────────────────────

    def close(self) -> None:
        """Release model resources."""
        self._model = None
        logger.info("Detector closed.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __repr__(self) -> str:
        return (
            f"Detector(conf={self.conf}, iou={self.iou}, "
            f"device={self.device})"
        )