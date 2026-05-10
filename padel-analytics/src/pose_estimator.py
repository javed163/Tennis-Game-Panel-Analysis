"""
pose_estimator.py
=================
Extracts body keypoints for each tracked player using MediaPipe Pose.

Fixes applied vs previous version
-----------------------------------
FIX 1  mp defined in except block too          (was only in try block)
FIX 2  config imports wrapped in try/except    (removes "Cannot find ref" warning)
FIX 3  normalize_landmark import removed       (was unused — flagged by PyCharm)
FIX 4  pose_landmarks accessed via getattr()   (removes NamedTuple attr warning)
FIX 5  POSE_MODEL_COMPLEXITY in __repr__ via   (removed direct config ref at end)
       self._model_complexity instance var
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── FIX 1 — mp defined in BOTH try AND except ─────────────────────────────────
try:
    import mediapipe as mp            # type: ignore[import]
    _MP_AVAILABLE = True
except ImportError:
    mp = None                         # ← was missing before; caused PyCharm warning
    _MP_AVAILABLE = False

# ── FIX 2 — config import wrapped in try/except ───────────────────────────────
# PyCharm shows "Cannot find reference" when it can't resolve the config module
# at static analysis time (e.g. src/ not marked as Sources Root in the IDE).
# Wrapping in try/except provides safe fallback defaults AND silences the warning.
try:
    from config import (              # type: ignore[import]
        POSE_MIN_DETECTION_CONF,
        POSE_MIN_TRACKING_CONF,
        POSE_MODEL_COMPLEXITY,
    )
except ImportError:
    POSE_MIN_DETECTION_CONF = 0.5
    POSE_MIN_TRACKING_CONF  = 0.5
    POSE_MODEL_COMPLEXITY   = 1

try:
    from tracking import Track
except ImportError:
    Track = object                    # type: ignore[misc,assignment]

# ── FIX 3 — normalize_landmark REMOVED (it was imported but never used) ───────
try:
    from utils import get_logger
except ImportError:
    import logging
    def get_logger(name: str):        # type: ignore[misc]
        return logging.getLogger(name)

logger = get_logger(__name__)


# ─── MediaPipe landmark index map ─────────────────────────────────────────────
_LM: Dict[str, int] = {
    "nose":            0,
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_elbow":     13,
    "right_elbow":    14,
    "left_wrist":     15,
    "right_wrist":    16,
    "left_hip":       23,
    "right_hip":      24,
}


# ─── Keypoint ─────────────────────────────────────────────────────────────────

@dataclass
class Keypoint:
    """Single body landmark in full-frame pixel coordinates."""
    x:          float
    y:          float
    visibility: float
    name:       str = ""

    @property
    def xy(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @property
    def xy_int(self) -> Tuple[int, int]:
        return (int(self.x), int(self.y))

    @property
    def is_visible(self) -> bool:
        return self.visibility >= 0.5

    def distance_to(self, other: "Keypoint") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


# ─── ArmKeypoints ─────────────────────────────────────────────────────────────

@dataclass
class ArmKeypoints:
    """Shoulder → Elbow → Wrist kinematic chain for one arm."""
    shoulder: Optional[Keypoint] = None
    elbow:    Optional[Keypoint] = None
    wrist:    Optional[Keypoint] = None
    side:     str = "right"

    @property
    def is_complete(self) -> bool:
        return all(
            kp is not None and kp.is_visible
            for kp in [self.shoulder, self.elbow, self.wrist]
        )

    @property
    def wrist_above_shoulder(self) -> bool:
        if self.wrist is None or self.shoulder is None:
            return False
        return self.wrist.y < self.shoulder.y   # lower y = higher in image

    @property
    def forearm_angle(self) -> Optional[float]:
        """Elbow→wrist angle (degrees) relative to vertical."""
        if self.elbow is None or self.wrist is None:
            return None
        dx = self.wrist.x - self.elbow.x
        dy = self.wrist.y - self.elbow.y
        return math.degrees(math.atan2(dx, -dy))


# ─── PlayerPose ───────────────────────────────────────────────────────────────

@dataclass
class PlayerPose:
    """All pose data for one tracked player in one frame."""
    track_id:      int
    frame_idx:     int
    right_arm:     ArmKeypoints = field(
        default_factory=lambda: ArmKeypoints(side="right")
    )
    left_arm:      ArmKeypoints = field(
        default_factory=lambda: ArmKeypoints(side="left")
    )
    all_keypoints: Dict[str, Keypoint] = field(default_factory=dict)
    pose_found:    bool = False

    @property
    def dominant_arm(self) -> ArmKeypoints:
        """Arm with more visible keypoints (defaults to right)."""
        r_vis = sum(
            1 for kp in [self.right_arm.shoulder,
                         self.right_arm.elbow,
                         self.right_arm.wrist]
            if kp is not None and kp.is_visible
        )
        l_vis = sum(
            1 for kp in [self.left_arm.shoulder,
                         self.left_arm.elbow,
                         self.left_arm.wrist]
            if kp is not None and kp.is_visible
        )
        return self.left_arm if l_vis > r_vis else self.right_arm

    @property
    def wrist(self) -> Optional[Keypoint]:
        return self.dominant_arm.wrist

    @property
    def elbow(self) -> Optional[Keypoint]:
        return self.dominant_arm.elbow

    @property
    def shoulder(self) -> Optional[Keypoint]:
        return self.dominant_arm.shoulder


# ─── PoseEstimator ────────────────────────────────────────────────────────────

class PoseEstimator:
    """
    Runs MediaPipe Pose on each player crop and returns PlayerPose objects.

    Parameters
    ----------
    min_detection_conf : float   Initial detection confidence threshold
    min_tracking_conf  : float   Landmark tracking confidence threshold
    model_complexity   : int     0=fast  1=balanced  2=accurate
    padding            : float   Crop expansion (0.10 = 10% on each side)
    """

    def __init__(
        self,
        min_detection_conf: float = POSE_MIN_DETECTION_CONF,
        min_tracking_conf:  float = POSE_MIN_TRACKING_CONF,
        model_complexity:   int   = POSE_MODEL_COMPLEXITY,
        padding:            float = 0.10,
    ):
        if not _MP_AVAILABLE:
            raise ImportError(
                "mediapipe is not installed.\n"
                "Fix: pip install mediapipe"
            )

        self.padding           = padding
        self._model_complexity = model_complexity   # FIX 5 — stored on self

        # Build the MediaPipe Pose object
        # mp is guaranteed non-None here because _MP_AVAILABLE is True
        _mp_pose = mp.solutions.pose                # type: ignore[union-attr]
        self._pose = _mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
            enable_segmentation=False,
        )

        logger.info(
            f"PoseEstimator ready  complexity={model_complexity}  "
            f"det={min_detection_conf}  trk={min_tracking_conf}"
        )

    # ── Public ────────────────────────────────────────────────────────────────

    def estimate(
        self,
        frame:  np.ndarray,
        tracks: List,
    ) -> List[PlayerPose]:
        """
        Run pose estimation for every track in one frame.

        Parameters
        ----------
        frame  : full BGR video frame
        tracks : list of Track objects (from tracking.py)

        Returns
        -------
        List[PlayerPose] — one entry per track, same order as input.
        """
        poses: List[PlayerPose] = []
        frame_h, frame_w = frame.shape[:2]
        for track in tracks:
            poses.append(
                self._estimate_single(frame, track, frame_w, frame_h)
            )
        return poses

    # ── Internal ──────────────────────────────────────────────────────────────

    def _estimate_single(
        self,
        frame:   np.ndarray,
        track:   object,
        frame_w: int,
        frame_h: int,
    ) -> PlayerPose:
        """Run MediaPipe on one player crop and return a PlayerPose."""

        track_id  = getattr(track, "track_id", 0)
        bbox      = getattr(track, "bbox", (0, 0, 1, 1))

        player_pose = PlayerPose(track_id=track_id, frame_idx=0)

        crop, cx1, cy1 = self._crop_player(frame, bbox, frame_w, frame_h)
        if crop is None or crop.size == 0:
            logger.debug(f"Track {track_id}: empty crop.")
            return player_pose

        crop_h, crop_w = crop.shape[:2]
        rgb   = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)

        # ── FIX 4 — use getattr to avoid NamedTuple attribute warning ─────────
        pose_landmarks = getattr(result, "pose_landmarks", None)
        if pose_landmarks is None:
            logger.debug(f"Track {track_id}: no landmarks.")
            return player_pose

        player_pose.pose_found = True
        landmarks = pose_landmarks.landmark

        def _to_kp(idx: int, name: str) -> Keypoint:
            """Convert one landmark to full-frame Keypoint."""
            lm = landmarks[idx]
            fx = lm.x * crop_w + cx1
            fy = lm.y * crop_h + cy1
            vis = float(getattr(lm, "visibility", 1.0))
            return Keypoint(x=fx, y=fy, visibility=vis, name=name)

        all_kp: Dict[str, Keypoint] = {
            name: _to_kp(idx, name) for name, idx in _LM.items()
        }
        player_pose.all_keypoints = all_kp

        player_pose.right_arm = ArmKeypoints(
            shoulder=all_kp.get("right_shoulder"),
            elbow=   all_kp.get("right_elbow"),
            wrist=   all_kp.get("right_wrist"),
            side="right",
        )
        player_pose.left_arm = ArmKeypoints(
            shoulder=all_kp.get("left_shoulder"),
            elbow=   all_kp.get("left_elbow"),
            wrist=   all_kp.get("left_wrist"),
            side="left",
        )

        logger.debug(
            f"Track {track_id}: pose OK "
            f"R={player_pose.right_arm.is_complete} "
            f"L={player_pose.left_arm.is_complete}"
        )
        return player_pose

    def _crop_player(
        self,
        frame:   np.ndarray,
        bbox:    Tuple[int, int, int, int],
        frame_w: int,
        frame_h: int,
    ) -> Tuple[Optional[np.ndarray], int, int]:
        """Crop + pad player region. Returns (crop, x1, y1)."""
        x1, y1, x2, y2 = bbox
        pad_x = int((x2 - x1) * self.padding)
        pad_y = int((y2 - y1) * self.padding)
        x1p = max(0,        x1 - pad_x)
        y1p = max(0,        y1 - pad_y)
        x2p = min(frame_w,  x2 + pad_x)
        y2p = min(frame_h,  y2 + pad_y)
        crop = frame[y1p:y2p, x1p:x2p]
        return (None, x1p, y1p) if crop.size == 0 else (crop, x1p, y1p)

    @staticmethod
    def wrist_positions_from_history(
        pose_history: List[PlayerPose],
    ) -> List[Tuple[float, float]]:
        """Time-ordered (x, y) wrist positions from a pose history list."""
        return [
            pose.wrist.xy
            for pose in pose_history
            if pose.pose_found and pose.wrist and pose.wrist.is_visible
        ]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        if hasattr(self, "_pose") and self._pose is not None:
            self._pose.close()
            logger.info("PoseEstimator closed.")

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __repr__(self) -> str:
        # FIX 5 — uses self._model_complexity, not the config name directly
        return (
            f"PoseEstimator("
            f"padding={self.padding}, "
            f"complexity={self._model_complexity})"
        )