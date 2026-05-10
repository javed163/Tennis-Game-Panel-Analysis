"""
visualizer.py
=============
Draws all visual overlays onto raw video frames using OpenCV:

  - Player bounding boxes  (colour-coded by track ID)
  - Stable player ID labels
  - Skeleton keypoints     (shoulder / elbow / wrist)
  - Shot type label        (shown for N frames after each shot event)
  - Ball bounding box
  - HUD panel              (shot counts, FPS, frame number)
  - Shot flash effect      (brief colour flash on new shot)

Public API
----------
    viz = Visualizer(frame_w=1280, frame_h=720)
    annotated = viz.draw(frame, detection_result, tracks, poses,
                         new_events, shot_counts, frame_idx, fps)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    BBOX_THICKNESS,
    FONT_SCALE,
    FONT_THICKNESS,
    COLOURS,
)
from detection import DetectionResult
from tracking import Track
from pose_estimator import PlayerPose
from shot_classifier import ShotEvent, ShotType
from utils import get_logger

logger = get_logger(__name__)


# ─── Colour palette per track ID ─────────────────────────────────────────────
# Cycles through distinct BGR colours so each player gets a unique colour
_TRACK_COLOURS = [
    (0,   200, 100),   # green
    (255, 100,   0),   # blue
    (0,   165, 255),   # orange
    (180,   0, 255),   # purple
    (0,   255, 255),   # yellow
    (255,   0, 180),   # pink
]

def _track_colour(track_id: int) -> Tuple[int, int, int]:
    return _TRACK_COLOURS[track_id % len(_TRACK_COLOURS)]


# ─── Shot type colours (BGR) ──────────────────────────────────────────────────
_SHOT_COLOURS_BGR = {
    ShotType.FOREHAND: (255, 100,   0),   # blue
    ShotType.BACKHAND: (0,   100, 255),   # red-orange
    ShotType.SMASH:    (0,     0, 255),   # red
    ShotType.UNKNOWN:  (200, 200, 200),   # grey
}

# ─── Skeleton connections (MediaPipe landmark name pairs) ─────────────────────
_SKELETON_CONNECTIONS = [
    ("left_shoulder",  "left_elbow"),
    ("left_elbow",     "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
    ("left_shoulder",  "right_shoulder"),
]


class Visualizer:
    """
    Draws all visual overlays onto a copy of each video frame.

    Parameters
    ----------
    frame_w         : int    Frame width  in pixels
    frame_h         : int    Frame height in pixels
    shot_label_dur  : int    How many frames to show the shot label after
                             a shot event is emitted
    show_skeleton   : bool   Draw skeleton keypoints / connections
    show_hud        : bool   Draw the HUD info panel
    show_ball       : bool   Draw ball bounding box
    hud_alpha       : float  Transparency of the HUD background (0–1)
    """

    def __init__(
        self,
        frame_w:        int   = 1280,
        frame_h:        int   = 720,
        shot_label_dur: int   = 40,
        show_skeleton:  bool  = True,
        show_hud:       bool  = True,
        show_ball:      bool  = True,
        hud_alpha:      float = 0.55,
    ):
        self.frame_w        = frame_w
        self.frame_h        = frame_h
        self.shot_label_dur = shot_label_dur
        self.show_skeleton  = show_skeleton
        self.show_hud       = show_hud
        self.show_ball      = show_ball
        self.hud_alpha      = hud_alpha

        # {track_id: (shot_type, frames_remaining)}
        self._active_labels: Dict[int, Tuple[str, int]] = {}

        # Track the last known shot per player for the HUD
        self._last_shot: Dict[int, str] = {}

        logger.info(
            f"Visualizer ready "
            f"({frame_w}x{frame_h}, "
            f"skeleton={show_skeleton}, hud={show_hud})"
        )

    # ── Public: draw ──────────────────────────────────────────────────────────

    def draw(
        self,
        frame:          np.ndarray,
        det_result:     DetectionResult,
        tracks:         List[Track],
        poses:          List[PlayerPose],
        new_events:     List[ShotEvent],
        shot_counts:    Dict[int, Dict[str, int]],
        frame_idx:      int,
        fps:            float = 30.0,
    ) -> np.ndarray:
        """
        Annotate one frame and return the annotated copy.

        Parameters
        ----------
        frame        : original BGR frame (not modified in-place)
        det_result   : DetectionResult from detection.py
        tracks       : active Track objects from tracking.py
        poses        : PlayerPose objects from pose_estimator.py
        new_events   : shot events emitted THIS frame
        shot_counts  : {track_id: {shot_type: count}} from analytics
        frame_idx    : current frame number
        fps          : video fps (for HUD display)

        Returns
        -------
        np.ndarray — annotated BGR frame
        """
        canvas = frame.copy()

        # ── Register new shot events (start label timers) ──────────────────
        self._register_events(new_events)

        # ── Build pose lookup {track_id: PlayerPose} ──────────────────────
        pose_map: Dict[int, PlayerPose] = {
            p.track_id: p for p in poses
        }

        # ── Draw ball ─────────────────────────────────────────────────────
        if self.show_ball and det_result.has_ball:
            for ball in det_result.balls:
                self._draw_ball(canvas, ball.bbox)

        # ── Draw players ──────────────────────────────────────────────────
        for track in tracks:
            colour = _track_colour(track.track_id)
            pose   = pose_map.get(track.track_id)

            # Bounding box
            self._draw_bbox(canvas, track.bbox, colour, track.track_id,
                            track.confidence)

            # Skeleton
            if self.show_skeleton and pose and pose.pose_found:
                self._draw_skeleton(canvas, pose)

            # Shot label (shown for shot_label_dur frames)
            label_info = self._active_labels.get(track.track_id)
            if label_info:
                shot_type, remaining = label_info
                if remaining > 0:
                    self._draw_shot_label(
                        canvas, track.bbox, shot_type, remaining
                    )
                    # Flash effect on the bbox when shot is fresh
                    if remaining > self.shot_label_dur - 8:
                        self._draw_flash(canvas, track.bbox, shot_type)

        # ── Tick down active label timers ─────────────────────────────────
        self._tick_labels()

        # ── HUD panel ─────────────────────────────────────────────────────
        if self.show_hud:
            self._draw_hud(canvas, shot_counts, frame_idx, fps)

        return canvas

    # ── Event registration ────────────────────────────────────────────────────

    def _register_events(self, events: List[ShotEvent]) -> None:
        """Start a label timer for each new shot event."""
        for event in events:
            self._active_labels[event.track_id] = (
                event.shot_type,
                self.shot_label_dur,
            )
            self._last_shot[event.track_id] = event.shot_type
            logger.debug(
                f"Label started: player={event.track_id} "
                f"type={event.shot_type}"
            )

    def _tick_labels(self) -> None:
        """Decrement all active label timers by one frame."""
        to_remove = []
        for tid, (stype, remaining) in self._active_labels.items():
            new_remaining = remaining - 1
            if new_remaining <= 0:
                to_remove.append(tid)
            else:
                self._active_labels[tid] = (stype, new_remaining)
        for tid in to_remove:
            del self._active_labels[tid]

    # ── Drawing primitives ────────────────────────────────────────────────────

    def _draw_bbox(
        self,
        canvas:     np.ndarray,
        bbox:       Tuple[int, int, int, int],
        colour:     Tuple[int, int, int],
        track_id:   int,
        confidence: float,
    ) -> None:
        """Draw a bounding box with a player ID label above it."""
        x1, y1, x2, y2 = bbox

        # Main rectangle
        cv2.rectangle(canvas, (x1, y1), (x2, y2),
                      colour, BBOX_THICKNESS, cv2.LINE_AA)

        # Corner accents (small L-shapes at each corner for style)
        corner_len = min(15, (x2 - x1) // 5, (y2 - y1) // 5)
        for (cx, cy, dx, dy) in [
            (x1, y1,  1,  1),
            (x2, y1, -1,  1),
            (x1, y2,  1, -1),
            (x2, y2, -1, -1),
        ]:
            cv2.line(canvas, (cx, cy),
                     (cx + dx * corner_len, cy), colour, 2, cv2.LINE_AA)
            cv2.line(canvas, (cx, cy),
                     (cx, cy + dy * corner_len), colour, 2, cv2.LINE_AA)

        # Label background pill
        label     = f"P{track_id}  {confidence:.0%}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS
        )
        pad = 4
        lx1 = x1
        ly1 = max(0, y1 - th - baseline - pad * 2)
        lx2 = x1 + tw + pad * 2
        ly2 = y1

        # Semi-transparent background
        sub    = canvas[ly1:ly2, lx1:lx2]
        if sub.size > 0:
            bg = np.full_like(sub, colour)
            cv2.addWeighted(bg, 0.7, sub, 0.3, 0, sub)
            canvas[ly1:ly2, lx1:lx2] = sub

        cv2.putText(
            canvas, label,
            (lx1 + pad, ly2 - baseline - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE, (255, 255, 255),
            FONT_THICKNESS, cv2.LINE_AA,
        )

    def _draw_ball(
        self,
        canvas: np.ndarray,
        bbox:   Tuple[int, int, int, int],
    ) -> None:
        """Draw the ball bounding box as a circle with a cross-hair."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        r  = max(6, (x2 - x1) // 2)
        colour = COLOURS["ball"]

        cv2.circle(canvas, (cx, cy), r, colour, 2, cv2.LINE_AA)
        # Cross-hair
        cv2.line(canvas, (cx - r - 4, cy), (cx + r + 4, cy),
                 colour, 1, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy - r - 4), (cx, cy + r + 4),
                 colour, 1, cv2.LINE_AA)

    def _draw_skeleton(
        self,
        canvas: np.ndarray,
        pose:   PlayerPose,
    ) -> None:
        """Draw arm skeleton lines and joint circles."""
        kps = pose.all_keypoints

        # ── Connections ───────────────────────────────────────────────────
        for name_a, name_b in _SKELETON_CONNECTIONS:
            kp_a = kps.get(name_a)
            kp_b = kps.get(name_b)
            if kp_a and kp_b and kp_a.is_visible and kp_b.is_visible:
                cv2.line(
                    canvas,
                    kp_a.xy_int, kp_b.xy_int,
                    (220, 220, 220), 2, cv2.LINE_AA,
                )

        # ── Joint circles ─────────────────────────────────────────────────
        joint_colours = {
            "left_shoulder":  (100, 255, 100),
            "right_shoulder": (100, 255, 100),
            "left_elbow":     (100, 200, 255),
            "right_elbow":    (100, 200, 255),
            "left_wrist":     (255, 200, 100),
            "right_wrist":    (255, 200, 100),
        }
        for name, colour in joint_colours.items():
            kp = kps.get(name)
            if kp and kp.is_visible:
                cv2.circle(canvas, kp.xy_int, 5,
                           colour, -1, cv2.LINE_AA)
                cv2.circle(canvas, kp.xy_int, 5,
                           (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_shot_label(
        self,
        canvas:    np.ndarray,
        bbox:      Tuple[int, int, int, int],
        shot_type: str,
        remaining: int,
    ) -> None:
        """
        Draw a shot-type label below the player bounding box.
        Fades out as 'remaining' approaches zero.
        """
        x1, y1, x2, y2 = bbox
        colour = _SHOT_COLOURS_BGR.get(shot_type, (200, 200, 200))

        # Fade alpha from 1.0 → 0.3 over the label duration
        alpha = 0.3 + 0.7 * (remaining / self.shot_label_dur)

        label = shot_type.upper()
        scale = FONT_SCALE * 1.2
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, scale, FONT_THICKNESS
        )

        pad  = 6
        lx   = x1 + (x2 - x1 - tw) // 2 - pad
        ly1_ = y2 + 6
        ly2_ = y2 + th + baseline + pad * 2 + 6

        lx  = max(0, min(lx, self.frame_w - tw - pad * 2))
        ly2_ = min(ly2_, self.frame_h)

        # Background
        sub = canvas[ly1_:ly2_, lx: lx + tw + pad * 2]
        if sub.size > 0:
            bg = np.full_like(sub, colour)
            cv2.addWeighted(bg, alpha * 0.8, sub, 1 - alpha * 0.8, 0, sub)
            canvas[ly1_:ly2_, lx: lx + tw + pad * 2] = sub

        # Text
        text_colour = tuple(
            int(c * alpha + 255 * (1 - alpha)) for c in (255, 255, 255)
        )
        cv2.putText(
            canvas, label,
            (lx + pad, ly2_ - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale, text_colour,
            FONT_THICKNESS, cv2.LINE_AA,
        )

    def _draw_flash(
        self,
        canvas:    np.ndarray,
        bbox:      Tuple[int, int, int, int],
        shot_type: str,
    ) -> None:
        """Briefly tint the player region with the shot colour."""
        x1, y1, x2, y2 = bbox
        colour = _SHOT_COLOURS_BGR.get(shot_type, (200, 200, 200))
        roi    = canvas[y1:y2, x1:x2]
        if roi.size == 0:
            return
        overlay = np.full_like(roi, colour)
        cv2.addWeighted(overlay, 0.25, roi, 0.75, 0, roi)
        canvas[y1:y2, x1:x2] = roi

    # ── HUD panel ─────────────────────────────────────────────────────────────

    def _draw_hud(
        self,
        canvas:      np.ndarray,
        shot_counts: Dict[int, Dict[str, int]],
        frame_idx:   int,
        fps:         float,
    ) -> None:
        """
        Draw a semi-transparent HUD panel in the top-right corner showing:
          - Frame number
          - FPS
          - Per-player shot totals
          - Last shot type per player
        """
        lines = [
            f"Frame : {frame_idx:05d}",
            f"FPS   : {fps:.1f}",
            "─" * 18,
        ]

        for tid in sorted(shot_counts.keys()):
            counts   = shot_counts[tid]
            total    = counts.get("total", 0)
            fh       = counts.get(ShotType.FOREHAND, 0)
            bh       = counts.get(ShotType.BACKHAND, 0)
            sm       = counts.get(ShotType.SMASH,    0)
            last     = self._last_shot.get(tid, "—")
            lines += [
                f"P{tid}: {total} shots",
                f"  FH={fh} BH={bh} SM={sm}",
                f"  Last: {last}",
            ]

        # Measure panel size
        font       = cv2.FONT_HERSHEY_SIMPLEX
        fscale     = 0.48
        fthick     = 1
        line_h     = 20
        max_w      = max(
            cv2.getTextSize(ln, font, fscale, fthick)[0][0]
            for ln in lines
        ) + 20
        panel_h    = line_h * len(lines) + 16
        panel_x1   = self.frame_w - max_w - 12
        panel_y1   = 10
        panel_x2   = self.frame_w - 12
        panel_y2   = panel_y1 + panel_h

        # Clamp to frame bounds
        panel_x1 = max(0, panel_x1)
        panel_y2 = min(self.frame_h, panel_y2)

        # Semi-transparent background
        sub = canvas[panel_y1:panel_y2, panel_x1:panel_x2]
        if sub.size > 0:
            bg = np.zeros_like(sub)
            cv2.addWeighted(bg, self.hud_alpha, sub,
                            1 - self.hud_alpha, 0, sub)
            canvas[panel_y1:panel_y2, panel_x1:panel_x2] = sub

        # Border
        cv2.rectangle(
            canvas,
            (panel_x1, panel_y1), (panel_x2, panel_y2),
            (100, 100, 100), 1, cv2.LINE_AA,
        )

        # Text lines
        for i, line in enumerate(lines):
            colour = (180, 230, 180) if line.startswith("P") else (220, 220, 220)
            cv2.putText(
                canvas, line,
                (panel_x1 + 8, panel_y1 + 14 + i * line_h),
                font, fscale, colour, fthick, cv2.LINE_AA,
            )

    # ── Utility ───────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all active labels and last-shot memory."""
        self._active_labels.clear()
        self._last_shot.clear()

    def __repr__(self) -> str:
        return (
            f"Visualizer("
            f"{self.frame_w}x{self.frame_h}, "
            f"skeleton={self.show_skeleton}, "
            f"hud={self.show_hud})"
        )