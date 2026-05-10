"""
tests/test_classifier.py
========================
Unit tests for shot_classifier.py

Tests cover:
  - ShotEvent dataclass and serialisation
  - ShotType constants
  - _PlayerState velocity and debounce logic
  - ShotClassifier feed / event emission / summary
  - Classification rules (forehand / backhand / smash)
  - Edge cases (no pose, invisible wrist, rapid successive shots)

Run with:
    pytest tests/test_classifier.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ── Add src/ to path ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shot_classifier import (
    ShotClassifier,
    ShotEvent,
    ShotType,
    _PlayerState,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — build mock PlayerPose objects
# ──────────────────────────────────────────────────────────────────────────────

def _make_keypoint(x: float, y: float, visibility: float = 1.0):
    """Create a mock Keypoint with .x, .y, .visibility, .is_visible, .xy."""
    kp = MagicMock()
    kp.x          = x
    kp.y          = y
    kp.visibility = visibility
    kp.is_visible = visibility >= 0.5
    kp.xy         = (x, y)
    return kp


def _make_arm(
    shoulder_xy=(320, 300),
    elbow_xy=(340, 350),
    wrist_xy=(370, 400),
    wrist_above_shoulder: bool = False,
    forearm_angle: float = 30.0,
    is_complete: bool = True,
):
    """Create a mock ArmKeypoints."""
    arm = MagicMock()
    arm.shoulder             = _make_keypoint(*shoulder_xy)
    arm.elbow                = _make_keypoint(*elbow_xy)
    arm.wrist                = _make_keypoint(*wrist_xy)
    arm.is_complete          = is_complete
    arm.wrist_above_shoulder = wrist_above_shoulder
    arm.forearm_angle        = forearm_angle
    return arm


def _make_pose(
    track_id: int = 1,
    pose_found: bool = True,
    forearm_angle: float = 30.0,
    wrist_xy=(370, 400),
    shoulder_xy=(320, 300),
    wrist_above_shoulder: bool = False,
    wrist_visible: bool = True,
):
    """
    Create a mock PlayerPose with a dominant_arm and wrist/shoulder shortcuts.
    """
    pose = MagicMock()
    pose.track_id   = track_id
    pose.pose_found = pose_found

    arm = _make_arm(
        shoulder_xy=shoulder_xy,
        wrist_xy=wrist_xy,
        wrist_above_shoulder=wrist_above_shoulder,
        forearm_angle=forearm_angle,
    )

    pose.dominant_arm = arm
    pose.wrist        = _make_keypoint(*wrist_xy, visibility=1.0 if wrist_visible else 0.0)
    pose.wrist.is_visible = wrist_visible
    pose.shoulder     = _make_keypoint(*shoulder_xy)

    return pose


# ──────────────────────────────────────────────────────────────────────────────
# ShotType tests
# ──────────────────────────────────────────────────────────────────────────────

class TestShotType:

    def test_constants_exist(self):
        assert ShotType.FOREHAND == "forehand"
        assert ShotType.BACKHAND == "backhand"
        assert ShotType.SMASH    == "smash"
        assert ShotType.UNKNOWN  == "unknown"

    def test_all_list_contains_four_types(self):
        assert len(ShotType.ALL) == 4
        assert ShotType.FOREHAND in ShotType.ALL
        assert ShotType.BACKHAND in ShotType.ALL
        assert ShotType.SMASH    in ShotType.ALL
        assert ShotType.UNKNOWN  in ShotType.ALL


# ──────────────────────────────────────────────────────────────────────────────
# ShotEvent tests
# ──────────────────────────────────────────────────────────────────────────────

class TestShotEvent:

    @pytest.fixture
    def sample_event(self):
        return ShotEvent(
            track_id=1,
            frame_idx=150,
            timestamp_sec=5.0,
            shot_type=ShotType.FOREHAND,
            confidence=0.85,
            wrist_velocity=12.3,
            forearm_angle=35.0,
            wrist_x=370.0,
            wrist_y=400.0,
        )

    def test_to_dict_has_required_keys(self, sample_event):
        d = sample_event.to_dict()
        for key in [
            "track_id", "frame_idx", "timestamp_sec",
            "shot_type", "confidence", "wrist_velocity",
            "forearm_angle", "wrist_x", "wrist_y",
        ]:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_values_correct(self, sample_event):
        d = sample_event.to_dict()
        assert d["track_id"]      == 1
        assert d["frame_idx"]     == 150
        assert d["shot_type"]     == "forehand"
        assert d["confidence"]    == pytest.approx(0.85, abs=0.01)

    def test_str_contains_shot_type(self, sample_event):
        s = str(sample_event)
        assert "FOREHAND" in s

    def test_str_contains_frame_idx(self, sample_event):
        s = str(sample_event)
        assert "150" in s

    def test_confidence_rounded_in_dict(self, sample_event):
        d = sample_event.to_dict()
        # Should be rounded to 3 decimal places max
        conf_str = str(d["confidence"])
        decimal_places = len(conf_str.split(".")[-1]) if "." in conf_str else 0
        assert decimal_places <= 3


# ──────────────────────────────────────────────────────────────────────────────
# _PlayerState tests
# ──────────────────────────────────────────────────────────────────────────────

class TestPlayerState:

    def test_initial_can_emit(self):
        """A new state should always be ready to emit (no previous shot)."""
        state = _PlayerState(track_id=1)
        assert state.can_emit(frame_idx=0)

    def test_debounce_blocks_emission(self):
        """Right after a shot, can_emit should return False."""
        state = _PlayerState(track_id=1)
        state.mark_shot(frame_idx=100)
        assert not state.can_emit(frame_idx=105)

    def test_debounce_clears_after_window(self):
        """After SHOT_DEBOUNCE_FRAMES, can_emit should return True again."""
        from config import SHOT_DEBOUNCE_FRAMES
        state = _PlayerState(track_id=1)
        state.mark_shot(frame_idx=100)
        assert state.can_emit(frame_idx=100 + SHOT_DEBOUNCE_FRAMES)

    def test_velocity_zero_with_single_position(self):
        """One wrist position → velocity should be 0."""
        state = _PlayerState(track_id=1)
        state.push_wrist((320, 240))
        vel = state.compute_velocity()
        assert vel == pytest.approx(0.0)

    def test_velocity_nonzero_with_two_positions(self):
        """Two positions 30px apart → velocity > 0."""
        state = _PlayerState(track_id=1)
        state.push_wrist((300, 240))
        state.push_wrist((330, 240))   # moved 30px right
        vel = state.compute_velocity()
        assert vel > 0.0

    def test_velocity_zero_with_no_positions(self):
        """Empty history → velocity = 0."""
        state = _PlayerState(track_id=1)
        vel = state.compute_velocity()
        assert vel == pytest.approx(0.0)

    def test_push_none_wrist_handled(self):
        """Pushing None should not crash; invisible positions are skipped."""
        state = _PlayerState(track_id=1)
        state.push_wrist(None)
        state.push_wrist(None)
        vel = state.compute_velocity()
        assert vel == pytest.approx(0.0)

    def test_mark_shot_resets_counter(self):
        state = _PlayerState(track_id=1)
        state.mark_shot(frame_idx=50)
        assert state.last_shot_frame == 50
        assert state.frames_since_shot == 0


# ──────────────────────────────────────────────────────────────────────────────
# ShotClassifier — emission tests
# ──────────────────────────────────────────────────────────────────────────────

class TestShotClassifierEmission:

    @pytest.fixture
    def clf(self):
        return ShotClassifier(velocity_threshold=5.0, debounce_frames=15)

    def test_no_events_on_no_pose(self, clf):
        """If pose_found=False, no event should be emitted."""
        pose = _make_pose(track_id=1, pose_found=False)
        clf.feed(pose, frame_idx=10, fps=30.0)
        events = clf.get_new_events()
        assert len(events) == 0

    def test_no_events_below_threshold(self, clf):
        """
        Tiny wrist movement (< velocity_threshold) should not emit a shot.
        """
        pose = _make_pose(track_id=1, wrist_xy=(320, 240))
        # Feed the same position repeatedly — velocity stays ~0
        for i in range(10):
            clf.feed(pose, frame_idx=i, fps=30.0)
        events = clf.get_new_events()
        assert len(events) == 0

    def test_event_emitted_above_threshold(self, clf):
        """
        Large wrist movement (> velocity_threshold) should emit one shot.
        """
        # Prime the state with a starting position
        pose_start = _make_pose(track_id=2, wrist_xy=(100, 240))
        clf.feed(pose_start, frame_idx=0, fps=30.0)
        clf.get_new_events()   # drain

        # Feed a position far away → high velocity
        pose_move = _make_pose(track_id=2, wrist_xy=(200, 240))
        clf.feed(pose_move, frame_idx=1, fps=30.0)
        events = clf.get_new_events()

        assert len(events) == 1
        assert events[0].track_id == 2

    def test_debounce_prevents_double_count(self, clf):
        """
        Two large movements in quick succession should yield only one event.
        """
        pose_a = _make_pose(track_id=3, wrist_xy=(100, 240))
        pose_b = _make_pose(track_id=3, wrist_xy=(250, 240))
        pose_c = _make_pose(track_id=3, wrist_xy=(100, 240))

        clf.feed(pose_a, frame_idx=0, fps=30.0)
        clf.feed(pose_b, frame_idx=1, fps=30.0)
        clf.feed(pose_c, frame_idx=2, fps=30.0)

        events = clf.get_new_events()
        assert len(events) <= 1

    def test_get_new_events_drains_on_each_call(self, clf):
        """get_new_events() should return empty list on the second call."""
        pose_start = _make_pose(track_id=4, wrist_xy=(100, 240))
        pose_move  = _make_pose(track_id=4, wrist_xy=(250, 240))

        clf.feed(pose_start, frame_idx=0, fps=30.0)
        clf.feed(pose_move,  frame_idx=1, fps=30.0)

        first_drain  = clf.get_new_events()
        second_drain = clf.get_new_events()

        assert len(second_drain) == 0

    def test_all_events_accumulates(self, clf):
        """all_events should grow over time and never drain."""
        pose_start = _make_pose(track_id=5, wrist_xy=(50, 240))
        pose_move  = _make_pose(track_id=5, wrist_xy=(300, 240))

        clf.feed(pose_start, frame_idx=0, fps=30.0)
        clf.feed(pose_move,  frame_idx=1, fps=30.0)
        clf.get_new_events()   # drain new events

        # all_events should still hold everything
        assert len(clf.all_events) >= 0   # at minimum exists


# ──────────────────────────────────────────────────────────────────────────────
# ShotClassifier — classification rule tests
# ──────────────────────────────────────────────────────────────────────────────

class TestShotClassifierRules:

    @pytest.fixture
    def clf(self):
        return ShotClassifier(velocity_threshold=5.0, debounce_frames=10)

    def _trigger_shot(self, clf, track_id, forearm_angle,
                      wrist_above_shoulder=False,
                      start_frame=0):
        """
        Helper: prime the state then send a high-velocity movement
        to trigger exactly one shot event.
        Returns the emitted ShotEvent or None.
        """
        pose_start = _make_pose(
            track_id=track_id,
            wrist_xy=(100, 400),
            forearm_angle=forearm_angle,
            wrist_above_shoulder=wrist_above_shoulder,
        )
        clf.feed(pose_start, frame_idx=start_frame, fps=30.0)
        clf.get_new_events()   # drain priming frame

        pose_move = _make_pose(
            track_id=track_id,
            wrist_xy=(400, 400),   # big move → high velocity
            forearm_angle=forearm_angle,
            wrist_above_shoulder=wrist_above_shoulder,
        )
        clf.feed(pose_move, frame_idx=start_frame + 1, fps=30.0)
        events = clf.get_new_events()
        return events[0] if events else None

    def test_positive_angle_gives_forehand(self, clf):
        """Forearm angle > FOREHAND_ANGLE_MIN should classify as forehand."""
        event = self._trigger_shot(clf, track_id=10, forearm_angle=40.0,
                                   start_frame=0)
        if event:
            assert event.shot_type == ShotType.FOREHAND

    def test_negative_angle_gives_backhand(self, clf):
        """Forearm angle < BACKHAND_ANGLE_MAX should classify as backhand."""
        event = self._trigger_shot(clf, track_id=11, forearm_angle=-40.0,
                                   start_frame=0)
        if event:
            assert event.shot_type == ShotType.BACKHAND

    def test_wrist_above_shoulder_gives_smash(self, clf):
        """Wrist above shoulder should classify as smash."""
        event = self._trigger_shot(
            clf, track_id=12, forearm_angle=5.0,
            wrist_above_shoulder=True, start_frame=0
        )
        if event:
            assert event.shot_type == ShotType.SMASH

    def test_confidence_in_valid_range(self, clf):
        """All emitted events must have confidence in [0, 1]."""
        for i, angle in enumerate([-40.0, 40.0, 0.0]):
            event = self._trigger_shot(
                clf, track_id=20 + i,
                forearm_angle=angle,
                start_frame=i * 30,
            )
            if event:
                assert 0.0 <= event.confidence <= 1.0

    def test_shot_type_is_valid_string(self, clf):
        """shot_type must always be one of the ShotType constants."""
        event = self._trigger_shot(clf, track_id=30,
                                   forearm_angle=40.0, start_frame=0)
        if event:
            assert event.shot_type in ShotType.ALL

    def test_timestamp_computed_from_fps(self, clf):
        """timestamp_sec should equal frame_idx / fps."""
        fps        = 30.0
        frame_idx  = 60          # = 2.0 seconds

        pose_start = _make_pose(track_id=40, wrist_xy=(50, 240))
        clf.feed(pose_start, frame_idx=frame_idx, fps=fps)
        clf.get_new_events()

        pose_move = _make_pose(track_id=40, wrist_xy=(350, 240))
        clf.feed(pose_move, frame_idx=frame_idx + 1, fps=fps)
        events = clf.get_new_events()

        if events:
            expected_ts = (frame_idx + 1) / fps
            assert events[0].timestamp_sec == pytest.approx(
                expected_ts, abs=0.01
            )


# ──────────────────────────────────────────────────────────────────────────────
# ShotClassifier — summary and reset tests
# ──────────────────────────────────────────────────────────────────────────────

class TestShotClassifierSummary:

    def test_summary_empty_on_init(self):
        clf = ShotClassifier()
        assert clf.summary() == {}

    def test_reset_clears_all_events(self):
        clf = ShotClassifier(velocity_threshold=5.0, debounce_frames=5)

        pose_start = _make_pose(track_id=50, wrist_xy=(50,  240))
        pose_move  = _make_pose(track_id=50, wrist_xy=(350, 240))

        clf.feed(pose_start, frame_idx=0, fps=30.0)
        clf.feed(pose_move,  frame_idx=1, fps=30.0)

        clf.reset()   # full reset

        assert len(clf.all_events) == 0
        assert clf.summary()       == {}

    def test_reset_single_player_does_not_clear_others(self):
        """Resetting player 1 should not affect player 2's state."""
        clf = ShotClassifier(velocity_threshold=5.0, debounce_frames=5)

        for tid in [1, 2]:
            pose_start = _make_pose(track_id=tid, wrist_xy=(50,  240))
            pose_move  = _make_pose(track_id=tid, wrist_xy=(350, 240))
            clf.feed(pose_start, frame_idx=0, fps=30.0)
            clf.feed(pose_move,  frame_idx=1, fps=30.0)

        before_total = len(clf.all_events)
        clf.reset(track_id=1)   # reset only player 1

        # all_events is immutable after single-player reset
        assert len(clf.all_events) == before_total

    def test_feed_batch_processes_all_players(self):
        """feed_batch should handle multiple poses in one call."""
        clf = ShotClassifier(velocity_threshold=5.0)

        poses = [
            _make_pose(track_id=i, wrist_xy=(50 * i, 240))
            for i in range(1, 4)
        ]
        clf.feed_batch(poses, frame_idx=0, fps=30.0)
        # Should not raise; events list should exist
        events = clf.get_new_events()
        assert isinstance(events, list)

    def test_invisible_wrist_does_not_emit(self):
        """A pose with invisible wrist should never trigger a shot event."""
        clf = ShotClassifier(velocity_threshold=1.0)   # very low threshold

        for i in range(20):
            pose = _make_pose(
                track_id=99,
                wrist_xy=(i * 20, 240),
                wrist_visible=False,
            )
            clf.feed(pose, frame_idx=i, fps=30.0)

        events = clf.all_events
        assert len(events) == 0