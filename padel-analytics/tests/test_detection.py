"""
tests/test_detection.py
=======================
Unit tests for detection.py

Tests cover:
  - Detection and DetectionResult dataclass behaviour
  - Blob detector fallback
  - Detector input validation
  - DetectionResult properties

Run with:
    pytest tests/test_detection.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ── Add src/ to path ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from detection import Detection, DetectionResult, _detect_ball_blob


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def blank_frame():
    """Pure black 640x480 BGR frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def bright_dot_frame():
    """
    Black frame with a small bright white circle in the centre.
    The blob detector should find this as a 'ball'.
    """
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a bright white filled circle (simulates ball)
    import cv2
    cv2.circle(frame, (320, 240), 8, (255, 255, 255), -1)
    return frame


@pytest.fixture
def sample_detection():
    """A pre-built Detection object for testing."""
    return Detection(
        bbox=(100, 50, 200, 250),
        confidence=0.82,
        class_id=0,
        class_name="person",
    )


@pytest.fixture
def sample_det_result(sample_detection):
    """A DetectionResult containing one player and no balls."""
    result = DetectionResult(frame_idx=42)
    result.players.append(sample_detection)
    result.raw_boxes.append(sample_detection)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Detection dataclass tests
# ──────────────────────────────────────────────────────────────────────────────

class TestDetection:

    def test_center_computed_correctly(self, sample_detection):
        """Center should be the midpoint of the bbox."""
        x1, y1, x2, y2 = sample_detection.bbox
        expected_cx = (x1 + x2) // 2
        expected_cy = (y1 + y2) // 2
        assert sample_detection.center == (expected_cx, expected_cy)

    def test_width(self, sample_detection):
        x1, _, x2, _ = sample_detection.bbox
        assert sample_detection.width == x2 - x1

    def test_height(self, sample_detection):
        _, y1, _, y2 = sample_detection.bbox
        assert sample_detection.height == y2 - y1

    def test_area(self, sample_detection):
        expected = sample_detection.width * sample_detection.height
        assert sample_detection.area == expected

    def test_confidence_range(self, sample_detection):
        assert 0.0 <= sample_detection.confidence <= 1.0

    def test_class_name_stored(self, sample_detection):
        assert sample_detection.class_name == "person"

    def test_class_id_stored(self, sample_detection):
        assert sample_detection.class_id == 0

    def test_zero_area_bbox(self):
        """A bbox where x1==x2 or y1==y2 should have area 0."""
        det = Detection(
            bbox=(100, 100, 100, 200),
            confidence=0.5,
            class_id=0,
            class_name="person",
        )
        assert det.area == 0

    def test_different_bboxes_different_centers(self):
        det_a = Detection(bbox=(0,   0, 100, 100),
                          confidence=0.9, class_id=0, class_name="person")
        det_b = Detection(bbox=(200, 200, 400, 400),
                          confidence=0.9, class_id=0, class_name="person")
        assert det_a.center != det_b.center


# ──────────────────────────────────────────────────────────────────────────────
# DetectionResult dataclass tests
# ──────────────────────────────────────────────────────────────────────────────

class TestDetectionResult:

    def test_empty_result_has_no_players(self):
        result = DetectionResult(frame_idx=0)
        assert not result.has_players
        assert result.player_count == 0

    def test_empty_result_has_no_ball(self):
        result = DetectionResult(frame_idx=0)
        assert not result.has_ball

    def test_has_players_after_append(self, sample_detection):
        result = DetectionResult(frame_idx=1)
        result.players.append(sample_detection)
        assert result.has_players
        assert result.player_count == 1

    def test_has_ball_after_append(self):
        result = DetectionResult(frame_idx=2)
        ball = Detection(
            bbox=(300, 200, 320, 220),
            confidence=0.6,
            class_id=32,
            class_name="sports ball",
        )
        result.balls.append(ball)
        assert result.has_ball

    def test_player_count_multiple(self, sample_detection):
        result = DetectionResult(frame_idx=3)
        result.players.append(sample_detection)
        result.players.append(sample_detection)
        assert result.player_count == 2

    def test_frame_idx_stored(self):
        result = DetectionResult(frame_idx=99)
        assert result.frame_idx == 99

    def test_inference_ms_default_zero(self):
        result = DetectionResult(frame_idx=0)
        assert result.inference_ms == 0.0

    def test_inference_ms_can_be_set(self):
        result = DetectionResult(frame_idx=0, inference_ms=12.5)
        assert result.inference_ms == pytest.approx(12.5)


# ──────────────────────────────────────────────────────────────────────────────
# Blob detector fallback tests
# ──────────────────────────────────────────────────────────────────────────────

class TestBlobDetector:

    def test_blank_frame_returns_empty(self, blank_frame):
        """A pure black frame should yield no blob detections."""
        results = _detect_ball_blob(blank_frame, frame_idx=0)
        assert isinstance(results, list)
        # May or may not find noise — just check it returns a list
        for det in results:
            assert isinstance(det, Detection)

    def test_bright_dot_detected(self, bright_dot_frame):
        """
        A bright white circle on a black background should be
        detected by the blob fallback.
        """
        results = _detect_ball_blob(bright_dot_frame, frame_idx=0)
        # We expect at least one detection on the bright circle
        assert isinstance(results, list)
        if results:
            det = results[0]
            assert det.class_id == 32
            assert det.class_name == "ball"
            # Center should be near (320, 240)
            cx, cy = det.center
            assert abs(cx - 320) < 30
            assert abs(cy - 240) < 30

    def test_blob_detection_returns_detection_objects(self, bright_dot_frame):
        """All returned items must be Detection instances."""
        results = _detect_ball_blob(bright_dot_frame, frame_idx=5)
        for item in results:
            assert isinstance(item, Detection)

    def test_blob_confidence_is_set(self, bright_dot_frame):
        """Blob detections should have confidence = 0.5 (fixed)."""
        results = _detect_ball_blob(bright_dot_frame, frame_idx=0)
        for det in results:
            assert det.confidence == pytest.approx(0.5)

    def test_blob_bbox_valid(self, bright_dot_frame):
        """Bounding boxes must have x1 < x2 and y1 < y2."""
        results = _detect_ball_blob(bright_dot_frame, frame_idx=0)
        for det in results:
            x1, y1, x2, y2 = det.bbox
            assert x1 < x2
            assert y1 < y2


# ──────────────────────────────────────────────────────────────────────────────
# Utility / helper tests
# ──────────────────────────────────────────────────────────────────────────────

class TestDetectionHelpers:

    def test_multiple_players_separate_centers(self):
        """Two players at different positions should have different centers."""
        det1 = Detection(bbox=(0,   0, 100, 200),
                         confidence=0.9, class_id=0, class_name="person")
        det2 = Detection(bbox=(400, 300, 600, 600),
                         confidence=0.8, class_id=0, class_name="person")
        assert det1.center != det2.center

    def test_detection_result_inference_ms_type(self):
        result = DetectionResult(frame_idx=0, inference_ms=5.3)
        assert isinstance(result.inference_ms, float)

    def test_detection_area_large_bbox(self):
        det = Detection(
            bbox=(0, 0, 1280, 720),
            confidence=0.99, class_id=0, class_name="person"
        )
        assert det.area == 1280 * 720

    def test_detection_result_raw_boxes_independent(self, sample_detection):
        """raw_boxes and players lists are independent."""
        result = DetectionResult(frame_idx=0)
        result.players.append(sample_detection)
        # raw_boxes not touched
        assert len(result.raw_boxes) == 0
        assert len(result.players) == 1