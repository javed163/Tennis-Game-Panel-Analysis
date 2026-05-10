"""
conftest.py
===========
Pytest configuration file — lives at the project root.

Automatically adds src/ to sys.path before any test runs,
so every test file can do:
    from detection import Detector
without needing to repeat the sys.path hack.
"""

import sys
from pathlib import Path

# ── Add src/ to path for all tests ────────────────────────────────────────────
SRC_PATH = Path(__file__).resolve().parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
