"""
exporter.py
===========
Handles all file output for the Padel Analytics pipeline:

  - shots.json   — every ShotEvent + full match summary
  - shots.csv    — flat table of shot events (one row per shot)
  - summary.json — match-level statistics only
  - PNG charts   — shot count bar, timeline, heatmap, dashboard
  - Video        — annotated output mp4 via OpenCV VideoWriter

Public API
----------
    exporter = Exporter()

    # Video frame-by-frame
    exporter.open_video_writer(frame_w, frame_h, fps)
    exporter.write_frame(annotated_frame)
    exporter.close_video_writer()

    # End-of-pipeline exports
    exporter.export_json(shot_events, match_summary)
    exporter.export_csv(shot_events)
    exporter.export_charts(analytics)
    exporter.export_all(shot_events, match_summary, analytics)
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from config import (
    OUTPUT_JSON,
    OUTPUT_CSV,
    OUTPUT_VIDEO,
    OUTPUT_FOURCC,
    OUTPUTS_DIR,
)
from shot_classifier import ShotEvent
from utils import get_logger

logger = get_logger(__name__)


class Exporter:
    """
    Writes all pipeline outputs to disk.

    Parameters
    ----------
    output_dir  : Path   Root directory for all outputs.
                         Defaults to config.OUTPUTS_DIR.
    json_path   : Path   Override default JSON output path.
    csv_path    : Path   Override default CSV output path.
    video_path  : Path   Override default video output path.
    """

    def __init__(
        self,
        output_dir:  Optional[Path] = None,
        json_path:   Optional[Path] = None,
        csv_path:    Optional[Path] = None,
        video_path:  Optional[Path] = None,
    ):
        self.output_dir = Path(output_dir) if output_dir else OUTPUTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.json_path  = Path(json_path)  if json_path  else OUTPUT_JSON
        self.csv_path   = Path(csv_path)   if csv_path   else OUTPUT_CSV
        self.video_path = Path(video_path) if video_path else OUTPUT_VIDEO

        # Video writer state
        self._writer:      Optional[cv2.VideoWriter] = None
        self._frames_written: int = 0

        logger.info(f"Exporter initialised → {self.output_dir}")

    # ──────────────────────────────────────────────────────────────────────────
    # Video writer
    # ──────────────────────────────────────────────────────────────────────────

    def open_video_writer(
        self,
        frame_w: int,
        frame_h: int,
        fps:     float,
        path:    Optional[Path] = None,
    ) -> None:
        """
        Open an OpenCV VideoWriter for streaming annotated frames to disk.

        Parameters
        ----------
        frame_w : output frame width  (pixels)
        frame_h : output frame height (pixels)
        fps     : frames per second of the output video
        path    : override the default output video path
        """
        out_path = Path(path) if path else self.video_path
        fourcc   = cv2.VideoWriter_fourcc(*OUTPUT_FOURCC)

        self._writer = cv2.VideoWriter(
            str(out_path), fourcc, fps, (frame_w, frame_h)
        )

        if not self._writer.isOpened():
            logger.error(f"VideoWriter failed to open: {out_path}")
            self._writer = None
            return

        self._frames_written = 0
        logger.info(
            f"VideoWriter opened → {out_path} "
            f"({frame_w}x{frame_h} @ {fps:.1f}fps)"
        )

    def write_frame(self, frame: np.ndarray) -> None:
        """Write one annotated frame to the output video."""
        if self._writer is None or not self._writer.isOpened():
            return
        self._writer.write(frame)
        self._frames_written += 1

    def close_video_writer(self) -> None:
        """Flush and close the VideoWriter."""
        if self._writer and self._writer.isOpened():
            self._writer.release()
            logger.info(
                f"VideoWriter closed. "
                f"Frames written: {self._frames_written}"
            )
        self._writer = None

    # ──────────────────────────────────────────────────────────────────────────
    # JSON export
    # ──────────────────────────────────────────────────────────────────────────

    def export_json(
        self,
        shot_events:   List[ShotEvent],
        match_summary: dict,
        path:          Optional[Path] = None,
    ) -> Path:
        """
        Write the full shots.json file containing:
          - "meta"    : export timestamp, total shots, total players
          - "summary" : match-level statistics from analytics.match_summary()
          - "shots"   : list of every ShotEvent as a dict

        Returns the path written.
        """
        out_path = Path(path) if path else self.json_path

        payload = {
            "meta": {
                "exported_at":   time.strftime("%Y-%m-%dT%H:%M:%S"),
                "total_shots":   len(shot_events),
                "total_players": len({e.track_id for e in shot_events}),
                "generator":     "Padel Analytics Pipeline v1.0",
            },
            "summary": match_summary,
            "shots": [e.to_dict() for e in
                      sorted(shot_events, key=lambda e: e.frame_idx)],
        }

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            logger.info(
                f"JSON exported → {out_path} "
                f"({len(shot_events)} shots)"
            )
        except OSError as exc:
            logger.error(f"JSON export failed: {exc}")

        return out_path

    # ──────────────────────────────────────────────────────────────────────────
    # CSV export
    # ──────────────────────────────────────────────────────────────────────────

    def export_csv(
        self,
        shot_events: List[ShotEvent],
        path:        Optional[Path] = None,
    ) -> Path:
        """
        Write shots.csv — one row per shot event.

        Columns
        -------
        frame_idx, timestamp_sec, player_id, shot_type,
        confidence, wrist_velocity, forearm_angle, wrist_x, wrist_y
        """
        out_path = Path(path) if path else self.csv_path

        fieldnames = [
            "frame_idx",
            "timestamp_sec",
            "track_id",
            "shot_type",
            "confidence",
            "wrist_velocity",
            "forearm_angle",
            "wrist_x",
            "wrist_y",
        ]

        sorted_events = sorted(shot_events, key=lambda e: e.frame_idx)

        try:
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for event in sorted_events:
                    row = event.to_dict()
                    # Rename key to match fieldnames
                    row["track_id"] = row.pop("track_id", event.track_id)
                    # Write only the columns we want
                    writer.writerow({k: row.get(k, "") for k in fieldnames})

            logger.info(
                f"CSV exported → {out_path} "
                f"({len(shot_events)} rows)"
            )
        except OSError as exc:
            logger.error(f"CSV export failed: {exc}")

        return out_path

    # ──────────────────────────────────────────────────────────────────────────
    # Summary JSON (match-level only, no shot list)
    # ──────────────────────────────────────────────────────────────────────────

    def export_summary(
        self,
        match_summary: dict,
        path:          Optional[Path] = None,
    ) -> Path:
        """
        Write a lightweight summary.json with only match-level stats.
        Useful for quick inspection without loading the full shots list.
        """
        out_path = Path(path) if path else (
            self.output_dir / "summary.json"
        )

        payload = {
            "meta": {
                "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "generator":   "Padel Analytics Pipeline v1.0",
            },
            **match_summary,
        }

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            logger.info(f"Summary JSON exported → {out_path}")
        except OSError as exc:
            logger.error(f"Summary export failed: {exc}")

        return out_path

    # ──────────────────────────────────────────────────────────────────────────
    # Chart exports
    # ──────────────────────────────────────────────────────────────────────────

    def export_charts(self, analytics) -> Dict[str, Path]:
        """
        Save all analytics charts as PNG files.

        Parameters
        ----------
        analytics : Analytics instance from analytics.py

        Returns
        -------
        Dict mapping chart name → saved path
        """
        saved: Dict[str, Path] = {}

        charts = {
            "shot_counts":  (
                analytics.plot_shot_counts,
                self.output_dir / "chart_shot_counts.png",
            ),
            "timeline": (
                analytics.plot_shot_timeline,
                self.output_dir / "chart_timeline.png",
            ),
            "heatmap": (
                analytics.plot_heatmap,
                self.output_dir / "chart_heatmap.png",
            ),
            "dashboard": (
                analytics.plot_dashboard,
                self.output_dir / "chart_dashboard.png",
            ),
        }

        for name, (plot_fn, out_path) in charts.items():
            try:
                fig = plot_fn(save_path=str(out_path))
                if fig is not None:
                    saved[name] = out_path
                    logger.info(f"Chart '{name}' saved → {out_path}")
                    # Close figure to free memory
                    try:
                        import matplotlib.pyplot as plt
                        plt.close(fig)
                    except Exception:
                        pass
            except Exception as exc:
                logger.warning(f"Chart '{name}' failed: {exc}")

        return saved

    # ──────────────────────────────────────────────────────────────────────────
    # Convenience: export everything at once
    # ──────────────────────────────────────────────────────────────────────────

    def export_all(
        self,
        shot_events:   List[ShotEvent],
        match_summary: dict,
        analytics,
    ) -> Dict[str, Path]:
        """
        Run all exports in sequence:
          1. shots.json
          2. shots.csv
          3. summary.json
          4. All PNG charts

        Returns a dict of {name: path} for every file written.
        """
        logger.info("Running full export...")

        written: Dict[str, Path] = {}

        written["json"]    = self.export_json(shot_events, match_summary)
        written["csv"]     = self.export_csv(shot_events)
        written["summary"] = self.export_summary(match_summary)

        chart_paths = self.export_charts(analytics)
        written.update(chart_paths)

        logger.info(
            f"Export complete. "
            f"{len(written)} files written to {self.output_dir}"
        )
        self._print_export_report(written)

        return written

    # ──────────────────────────────────────────────────────────────────────────
    # Pretty-print export report
    # ──────────────────────────────────────────────────────────────────────────

    def _print_export_report(self, written: Dict[str, Path]) -> None:
        """Log a tidy summary of every file written."""
        separator = "─" * 52
        logger.info(separator)
        logger.info("  EXPORT REPORT")
        logger.info(separator)
        for name, path in written.items():
            try:
                size_kb = Path(path).stat().st_size / 1024
                logger.info(f"  {name:<14} {size_kb:>7.1f} KB   {path.name}")
            except OSError:
                logger.info(f"  {name:<14}  [file not found]  {path}")
        logger.info(separator)

    # ──────────────────────────────────────────────────────────────────────────
    # Context manager support
    # ──────────────────────────────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close_video_writer()

    def __repr__(self) -> str:
        return (
            f"Exporter("
            f"output_dir={self.output_dir}, "
            f"frames_written={self._frames_written})"
        )