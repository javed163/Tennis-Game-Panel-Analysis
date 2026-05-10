"""
tracking.py
===========
Multi-object tracker that assigns stable integer IDs to each detected
player across frames using ByteTrack (built into Ultralytics).

Without tracking, a player detected in frame 1 and frame 2 has no
guaranteed connection — YOLO just gives you boxes. ByteTrack links
those boxes over time so Player-1 stays Player-1 even when they move,
occlude each other, or briefly leave the frame.

Public API
----------
    tracker = Tracker()
    tracks  = tracker.update(detection_result, frame)   # List[Track]
    tracker.reset()
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import MAX_PLAYERS, SHOT_DEBOUNCE_FRAMES
from detection import Detection, DetectionResult
from utils import get_logger, euclidean_distance

logger = get_logger(__name__)


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Track:
    """
    Represents one tracked player across frames.

    Attributes
    ----------
    track_id    : stable integer assigned by the tracker
    bbox        : current bounding box (x1, y1, x2, y2)
    confidence  : detection confidence of the matched detection
    center      : (cx, cy) pixel center of the bbox
    age         : how many frames this track has existed
    hits        : how many frames it has been matched (not lost)
    is_confirmed: True once the track has been matched enough times
    history     : list of (cx, cy) centers for the last N frames
    """
    track_id:    int
    bbox:        Tuple[int, int, int, int]
    confidence:  float
    center:      Tuple[int, int]
    age:         int = 0
    hits:        int = 0
    is_confirmed: bool = False
    history:     List[Tuple[int, int]] = field(default_factory=list)

    # Maximum center history we keep per track
    MAX_HISTORY: int = field(default=60, init=False, repr=False)

    def update_history(self) -> None:
        """Append current center to history, capping at MAX_HISTORY."""
        self.history.append(self.center)
        if len(self.history) > self.MAX_HISTORY:
            self.history.pop(0)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def velocity(self) -> Tuple[float, float]:
        """
        Estimated (vx, vy) velocity in pixels/frame using the last
        two history entries. Returns (0, 0) if not enough history.
        """
        if len(self.history) < 2:
            return (0.0, 0.0)
        p1 = self.history[-2]
        p2 = self.history[-1]
        return (p2[0] - p1[0], p2[1] - p1[1])

    @property
    def speed(self) -> float:
        """Scalar speed in pixels/frame."""
        vx, vy = self.velocity
        return (vx ** 2 + vy ** 2) ** 0.5


# ─── Simple IoU helper ────────────────────────────────────────────────────────

def _iou(box_a: Tuple[int, int, int, int],
         box_b: Tuple[int, int, int, int]) -> float:
    """
    Intersection-over-Union between two bounding boxes.
    Both in (x1, y1, x2, y2) format.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


# ─── Tracker class ────────────────────────────────────────────────────────────

class Tracker:
    """
    Lightweight ByteTrack-style tracker implemented in pure Python/NumPy.

    Matching strategy
    -----------------
    1. For each new detection, compute IoU against all existing tracks.
    2. Greedily match the highest-IoU pairs (Hungarian-lite).
    3. Unmatched detections → new tentative tracks.
    4. Unmatched tracks → increment lost counter; remove if too old.

    Parameters
    ----------
    iou_threshold   : float  Minimum IoU to consider a match.
    max_lost        : int    Frames a track can go unmatched before deletion.
    min_hits        : int    Frames needed before a track is 'confirmed'.
    max_players     : int    Hard cap on simultaneous tracks (from config).
    """

    def __init__(
        self,
        iou_threshold: float = 0.30,
        max_lost:      int   = 30,
        min_hits:      int   = 3,
        max_players:   int   = MAX_PLAYERS,
    ):
        self.iou_threshold = iou_threshold
        self.max_lost      = max_lost
        self.min_hits      = min_hits
        self.max_players   = max_players

        self._tracks:   Dict[int, Track] = {}   # track_id → Track
        self._lost:     Dict[int, int]   = {}   # track_id → frames lost
        self._next_id:  int = 1

    # ── Public: update ────────────────────────────────────────────────────────

    def update(
        self,
        detection_result: DetectionResult,
        frame: Optional[np.ndarray] = None,
    ) -> List[Track]:
        """
        Feed new detections into the tracker and return all active tracks.

        Parameters
        ----------
        detection_result : DetectionResult from detection.py
        frame            : current BGR frame (unused here, kept for API compat)

        Returns
        -------
        List[Track] — only confirmed + recently-seen tracks.
        """
        detections = detection_result.players   # track persons only
        self._step(detections)
        return self.active_tracks

    # ── Internal: one tracking step ───────────────────────────────────────────

    def _step(self, detections: List[Detection]) -> None:
        """Core matching logic for one frame."""

        # ── Age all existing tracks ───────────────────────────────────────────
        for tid in list(self._tracks):
            self._tracks[tid].age += 1

        # ── Nothing to match against ──────────────────────────────────────────
        if not self._tracks:
            for det in detections[:self.max_players]:
                self._create_track(det)
            return

        track_ids  = list(self._tracks.keys())
        track_list = [self._tracks[tid] for tid in track_ids]

        matched_track_ids = set()
        matched_det_idxs  = set()

        # ── Build IoU matrix  (tracks × detections) ───────────────────────────
        iou_matrix = np.zeros((len(track_list), len(detections)), dtype=float)
        for ti, track in enumerate(track_list):
            for di, det in enumerate(detections):
                iou_matrix[ti, di] = _iou(track.bbox, det.bbox)

        # ── Greedy matching: best IoU first ───────────────────────────────────
        # Flatten, sort descending, pick non-conflicting pairs
        pairs = sorted(
            [(iou_matrix[ti, di], ti, di)
             for ti in range(len(track_list))
             for di in range(len(detections))],
            reverse=True,
        )

        for score, ti, di in pairs:
            if score < self.iou_threshold:
                break
            if ti in matched_track_ids or di in matched_det_idxs:
                continue
            # Accept this match
            tid = track_ids[ti]
            self._update_track(tid, detections[di])
            matched_track_ids.add(ti)
            matched_det_idxs.add(di)

        # ── Unmatched tracks → mark as lost ───────────────────────────────────
        for ti, tid in enumerate(track_ids):
            if ti not in matched_track_ids:
                self._lost[tid] = self._lost.get(tid, 0) + 1
                if self._lost[tid] > self.max_lost:
                    logger.debug(f"Track {tid} removed (lost > {self.max_lost}).")
                    del self._tracks[tid]
                    del self._lost[tid]

        # ── Unmatched detections → create new tracks ──────────────────────────
        total_tracks = len(self._tracks)
        for di, det in enumerate(detections):
            if di not in matched_det_idxs:
                if total_tracks < self.max_players:
                    self._create_track(det)
                    total_tracks += 1

    # ── Track lifecycle ───────────────────────────────────────────────────────

    def _create_track(self, det: Detection) -> None:
        """Initialise a new tentative track from a detection."""
        tid = self._next_id
        self._next_id += 1
        track = Track(
            track_id=tid,
            bbox=det.bbox,
            confidence=det.confidence,
            center=det.center,
            age=1,
            hits=1,
        )
        track.update_history()
        self._tracks[tid] = track
        self._lost[tid]   = 0
        logger.debug(f"New track created: ID={tid}")

    def _update_track(self, tid: int, det: Detection) -> None:
        """Update an existing track with a new matched detection."""
        track = self._tracks[tid]
        track.bbox       = det.bbox
        track.confidence = det.confidence
        track.center     = det.center
        track.hits      += 1
        track.is_confirmed = (track.hits >= self.min_hits)
        track.update_history()
        self._lost[tid] = 0   # reset lost counter on match

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def active_tracks(self) -> List[Track]:
        """
        Return tracks that are confirmed OR have been seen recently
        (hits >= 1 and not lost for more than 5 frames).
        Sorted by track_id for deterministic output.
        """
        return sorted(
            [t for tid, t in self._tracks.items()
             if t.is_confirmed or self._lost.get(tid, 0) <= 5],
            key=lambda t: t.track_id,
        )

    @property
    def track_count(self) -> int:
        return len(self.active_tracks)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_track(self, track_id: int) -> Optional[Track]:
        """Return a specific Track by ID, or None if not found."""
        return self._tracks.get(track_id)

    def reset(self) -> None:
        """Clear all tracks and reset the ID counter."""
        self._tracks.clear()
        self._lost.clear()
        self._next_id = 1
        logger.info("Tracker reset.")

    def track_centers(self) -> Dict[int, Tuple[int, int]]:
        """Return {track_id: (cx, cy)} for all active tracks."""
        return {t.track_id: t.center for t in self.active_tracks}

    def track_history(self, track_id: int) -> List[Tuple[int, int]]:
        """Return the center-point history for a given track ID."""
        track = self._tracks.get(track_id)
        return track.history if track else []

    def __repr__(self) -> str:
        return (
            f"Tracker(active={self.track_count}, "
            f"iou_thresh={self.iou_threshold}, "
            f"max_lost={self.max_lost})"
        )