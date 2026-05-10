"""
src/__init__.py
===============
Makes the src/ directory a proper Python package.
All modules can be imported directly when sys.path includes src/.

Exposed at package level for convenience:
    from src import Detector, Tracker, ShotClassifier ...
"""

# ── Version ───────────────────────────────────────────────────────────────────
__version__  = "1.0.0"
__author__   = "Padel Analytics"
__project__  = "Padel Game Analytics — Shot Classification System"

# ── Convenience re-exports ────────────────────────────────────────────────────
# These allow:  from src import Detector
# instead of:   from src.detection import Detector

from detection       import Detector, Detection, DetectionResult
from tracking        import Tracker, Track
from pose_estimator  import PoseEstimator, PlayerPose, ArmKeypoints, Keypoint
from shot_classifier import ShotClassifier, ShotEvent, ShotType
from analytics       import Analytics, Rally
from visualizer      import Visualizer
from exporter        import Exporter

__all__ = [
    # Detection
    "Detector", "Detection", "DetectionResult",
    # Tracking
    "Tracker", "Track",
    # Pose
    "PoseEstimator", "PlayerPose", "ArmKeypoints", "Keypoint",
    # Classification
    "ShotClassifier", "ShotEvent", "ShotType",
    # Analytics
    "Analytics", "Rally",
    # Visualizer
    "Visualizer",
    # Exporter
    "Exporter",
]