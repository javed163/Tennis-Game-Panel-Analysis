"""
shot_classifier.py
==================
Classifies padel shots (Forehand / Backhand / Smash) from a sequence
of PlayerPose objects using a rule-based engine.

Classification logic
--------------------
Each frame, we compute:
  1. Wrist velocity  — how fast the wrist is moving (pixels / frame)
  2. Forearm angle   — angle of elbow→wrist vector relative to vertical
  3. Wrist height    — is the wrist above the shoulder? (smash indicator)

A shot EVENT is emitted when:
  - Velocity crosses SHOT_VELOCITY_THRESHOLD  (swing detected)
  - A debounce window (SHOT_DEBOUNCE_FRAMES) has passed since the last shot
    (prevents double-counting one swing)

Shot type is decided from the forearm angle + height at peak velocity:

    wrist above shoulder                     → Smash / Serve
    forearm angle > FOREHAND_ANGLE_MIN       → Forehand
    forearm angle < BACKHAND_ANGLE_MAX       → Backhand
    otherwise                                → Unknown

Public API
----------
    clf = ShotClassifier()
    clf.feed(player_pose, frame_idx, fps)    # call every frame
    events = clf.get_new_events()            # drain emitted events
    clf.reset(track_id)                      # clear state for one player
    clf.all_events                           # list of all ShotEvent so far
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque

from config import (
    SHOT_VELOCITY_THRESHOLD,
    FOREHAND_ANGLE_MIN,
    BACKHAND_ANGLE_MAX,
    SMASH_HEIGHT_RATIO,
    SHOT_DEBOUNCE_FRAMES,
    VELOCITY_SMOOTH_WINDOW,
)
from pose_estimator import PlayerPose, Keypoint
from utils import get_logger, rolling_mean, timestamp_seconds

logger = get_logger(__name__)


# ─── Shot types ───────────────────────────────────────────────────────────────

class ShotType:
    FOREHAND = "forehand"
    BACKHAND = "backhand"
    SMASH    = "smash"
    UNKNOWN  = "unknown"

    ALL = [FOREHAND, BACKHAND, SMASH, UNKNOWN]


# ─── Shot event ───────────────────────────────────────────────────────────────

@dataclass
class ShotEvent:
    """
    A single classified shot event.

    Attributes
    ----------
    track_id        : player who hit the shot
    frame_idx       : frame number at peak wrist velocity
    timestamp_sec   : time in the video (seconds)
    shot_type       : ShotType string
    confidence      : rough confidence score 0–1
    wrist_velocity  : wrist speed at the moment of classification (px/frame)
    forearm_angle   : forearm angle (degrees) at moment of classification
    wrist_x, wrist_y: wrist pixel position at moment of classification
    """
    track_id:      int
    frame_idx:     int
    timestamp_sec: float
    shot_type:     str
    confidence:    float
    wrist_velocity: float
    forearm_angle:  float
    wrist_x:       float = 0.0
    wrist_y:       float = 0.0

    def to_dict(self) -> dict:
        return {
            "track_id":       self.track_id,
            "frame_idx":      self.frame_idx,
            "timestamp_sec":  self.timestamp_sec,
            "shot_type":      self.shot_type,
            "confidence":     round(self.confidence, 3),
            "wrist_velocity": round(self.wrist_velocity, 2),
            "forearm_angle":  round(self.forearm_angle, 2),
            "wrist_x":        round(self.wrist_x, 1),
            "wrist_y":        round(self.wrist_y, 1),
        }

    def __str__(self) -> str:
        return (
            f"[Frame {self.frame_idx:05d} | "
            f"t={self.timestamp_sec:.2f}s | "
            f"Player {self.track_id}] "
            f"{self.shot_type.upper():<10} "
            f"vel={self.wrist_velocity:.1f}px "
            f"angle={self.forearm_angle:.1f}°"
        )


# ─── Per-player state ─────────────────────────────────────────────────────────

@dataclass
class _PlayerState:
    """
    Internal rolling state kept per tracked player.

    We store a short history of wrist positions and velocities so we
    can smooth the velocity signal and detect the peak of a swing.
    """
    track_id:      int
    wrist_history: Deque[Optional[tuple]] = field(
        default_factory=lambda: deque(maxlen=30)
    )
    velocity_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=30)
    )
    last_shot_frame: int = -999          # frame of the most recent shot event
    frames_since_shot: int = 999         # convenience counter

    def push_wrist(self, wrist_xy: Optional[tuple]) -> None:
        self.wrist_history.append(wrist_xy)

    def push_velocity(self, v: float) -> None:
        self.velocity_history.append(v)

    def can_emit(self, frame_idx: int) -> bool:
        """True when debounce window has passed."""
        return (frame_idx - self.last_shot_frame) >= SHOT_DEBOUNCE_FRAMES

    def mark_shot(self, frame_idx: int) -> None:
        self.last_shot_frame = frame_idx
        self.frames_since_shot = 0

    def compute_velocity(self) -> float:
        """
        Compute smoothed wrist velocity from the last two visible
        wrist positions in the history buffer.
        Returns 0.0 if not enough data.
        """
        visible = [p for p in self.wrist_history if p is not None]
        if len(visible) < 2:
            return 0.0
        p1 = visible[-2]
        p2 = visible[-1]
        raw_vel = math.hypot(p2[0] - p1[0], p2[1] - p1[1])

        # Use rolling mean over velocity history for smoothing
        all_vels = list(self.velocity_history) + [raw_vel]
        smoothed = rolling_mean(all_vels, VELOCITY_SMOOTH_WINDOW)
        return smoothed[-1]


# ─── Shot Classifier ──────────────────────────────────────────────────────────

class ShotClassifier:
    """
    Stateful per-player shot classifier.

    Call .feed() once per frame for each player pose.
    Collect emitted ShotEvent objects via .get_new_events().

    Parameters
    ----------
    velocity_threshold  : float  Override config.SHOT_VELOCITY_THRESHOLD.
    debounce_frames     : int    Override config.SHOT_DEBOUNCE_FRAMES.
    """

    def __init__(
        self,
        velocity_threshold: float = SHOT_VELOCITY_THRESHOLD,
        debounce_frames:    int   = SHOT_DEBOUNCE_FRAMES,
    ):
        self.velocity_threshold = velocity_threshold
        self.debounce_frames    = debounce_frames

        self._states:     Dict[int, _PlayerState] = {}   # track_id → state
        self._all_events: List[ShotEvent] = []
        self._new_events: List[ShotEvent] = []           # drained by caller

    # ── Public: feed ──────────────────────────────────────────────────────────

    def feed(
        self,
        pose:      PlayerPose,
        frame_idx: int,
        fps:       float = 30.0,
    ) -> None:
        """
        Process one PlayerPose for one frame.

        Parameters
        ----------
        pose      : PlayerPose from pose_estimator.py
        frame_idx : current frame number
        fps       : video frame rate (used to compute timestamp)
        """
        state = self._get_or_create_state(pose.track_id)

        # ── Push wrist position (None if pose not found) ───────────────────
        if pose.pose_found and pose.wrist and pose.wrist.is_visible:
            state.push_wrist(pose.wrist.xy)
        else:
            state.push_wrist(None)
            state.push_velocity(0.0)
            return

        # ── Compute smoothed velocity ──────────────────────────────────────
        velocity = state.compute_velocity()
        state.push_velocity(velocity)

        logger.debug(
            f"Track {pose.track_id} | frame {frame_idx} | "
            f"vel={velocity:.2f} threshold={self.velocity_threshold}"
        )

        # ── Check if velocity crosses threshold AND debounce allows ────────
        if (velocity >= self.velocity_threshold
                and state.can_emit(frame_idx)):

            shot_type, confidence = self._classify_shot(pose, velocity)

            event = ShotEvent(
                track_id=pose.track_id,
                frame_idx=frame_idx,
                timestamp_sec=timestamp_seconds(frame_idx, fps),
                shot_type=shot_type,
                confidence=confidence,
                wrist_velocity=round(velocity, 2),
                forearm_angle=round(
                    pose.dominant_arm.forearm_angle or 0.0, 2
                ),
                wrist_x=pose.wrist.x if pose.wrist else 0.0,
                wrist_y=pose.wrist.y if pose.wrist else 0.0,
            )

            state.mark_shot(frame_idx)
            self._all_events.append(event)
            self._new_events.append(event)

            logger.info(str(event))

    # ── Public: feed_batch ────────────────────────────────────────────────────

    def feed_batch(
        self,
        poses:     List[PlayerPose],
        frame_idx: int,
        fps:       float = 30.0,
    ) -> None:
        """
        Process a list of PlayerPose objects (all players in one frame).
        Convenience wrapper around feed().
        """
        for pose in poses:
            self.feed(pose, frame_idx, fps)

    # ── Public: get_new_events ─────────────────────────────────────────────────

    def get_new_events(self) -> List[ShotEvent]:
        """
        Return and clear the list of events emitted since the last call.
        Call this every frame to drain new shots.
        """
        events = list(self._new_events)
        self._new_events.clear()
        return events

    # ── Public: all_events ────────────────────────────────────────────────────

    @property
    def all_events(self) -> List[ShotEvent]:
        """Complete list of all shot events detected so far."""
        return list(self._all_events)

    # ── Public: summary ───────────────────────────────────────────────────────

    def summary(self) -> Dict[int, Dict[str, int]]:
        """
        Return shot counts per player per shot type.

        Returns
        -------
        {
            track_id: {
                "forehand": N,
                "backhand": N,
                "smash":    N,
                "unknown":  N,
                "total":    N,
            },
            ...
        }
        """
        counts: Dict[int, Dict[str, int]] = defaultdict(
            lambda: {t: 0 for t in ShotType.ALL + ["total"]}
        )
        for event in self._all_events:
            counts[event.track_id][event.shot_type] += 1
            counts[event.track_id]["total"] += 1
        return dict(counts)

    # ── Public: reset ─────────────────────────────────────────────────────────

    def reset(self, track_id: Optional[int] = None) -> None:
        """
        Reset state.
        If track_id is given, reset only that player.
        If None, reset everything including all events.
        """
        if track_id is not None:
            self._states.pop(track_id, None)
        else:
            self._states.clear()
            self._all_events.clear()
            self._new_events.clear()
        logger.debug(f"ShotClassifier reset (track_id={track_id}).")

    # ── Classification logic ──────────────────────────────────────────────────

    def _classify_shot(
        self,
        pose:     PlayerPose,
        velocity: float,
    ) -> tuple[str, float]:
        """
        Decide shot type and a rough confidence score from pose features.

        Returns
        -------
        (shot_type: str, confidence: float)
        """
        arm   = pose.dominant_arm
        angle = arm.forearm_angle     # degrees, None if elbow/wrist missing

        # ── Feature 1: wrist above shoulder → Smash ───────────────────────
        if arm.wrist_above_shoulder:
            confidence = self._smash_confidence(velocity, arm)
            return ShotType.SMASH, confidence

        # ── Feature 2: forearm angle discriminates forehand vs backhand ────
        if angle is not None:
            if angle >= FOREHAND_ANGLE_MIN:
                confidence = self._angle_confidence(angle, ShotType.FOREHAND)
                return ShotType.FOREHAND, confidence

            if angle <= BACKHAND_ANGLE_MAX:
                confidence = self._angle_confidence(angle, ShotType.BACKHAND)
                return ShotType.BACKHAND, confidence

        # ── Fallback: velocity alone is strong but angle is ambiguous ──────
        # Use the wrist x-position relative to body center as a proxy
        if pose.shoulder and pose.wrist:
            wrist_right_of_shoulder = (
                pose.wrist.x > pose.shoulder.x
            )
            if wrist_right_of_shoulder:
                return ShotType.FOREHAND, 0.55
            else:
                return ShotType.BACKHAND, 0.55

        return ShotType.UNKNOWN, 0.40

    # ── Confidence helpers ────────────────────────────────────────────────────

    @staticmethod
    def _smash_confidence(velocity: float, arm) -> float:
        """
        Higher velocity and more extreme wrist-above-shoulder = more confident.
        Clipped to [0.55, 0.95].
        """
        base = min(velocity / (SHOT_VELOCITY_THRESHOLD * 3), 1.0)
        # Extra boost if shoulder is also visible and clearly below wrist
        if arm.shoulder and arm.wrist:
            height_diff = arm.shoulder.y - arm.wrist.y   # positive = wrist higher
            height_bonus = min(height_diff / 100.0, 0.2)
        else:
            height_bonus = 0.0
        return round(min(max(0.55 + base * 0.3 + height_bonus, 0.55), 0.95), 3)

    @staticmethod
    def _angle_confidence(angle: float, shot_type: str) -> float:
        """
        Confidence based on how far the angle is from the threshold boundary.
        Clips to [0.55, 0.92].
        """
        if shot_type == ShotType.FOREHAND:
            # Perfect forehand ≈ 45°; boundary at FOREHAND_ANGLE_MIN
            deviation = abs(angle - 45.0)
        else:
            # Perfect backhand ≈ -45°; boundary at BACKHAND_ANGLE_MAX
            deviation = abs(angle + 45.0)

        # The closer to 0 deviation, the more confident
        confidence = max(0.55, 0.92 - deviation / 200.0)
        return round(confidence, 3)

    # ── State management ──────────────────────────────────────────────────────

    def _get_or_create_state(self, track_id: int) -> _PlayerState:
        if track_id not in self._states:
            self._states[track_id] = _PlayerState(track_id=track_id)
        return self._states[track_id]

    def __repr__(self) -> str:
        return (
            f"ShotClassifier("
            f"events={len(self._all_events)}, "
            f"players={len(self._states)}, "
            f"vel_threshold={self.velocity_threshold})"
        )