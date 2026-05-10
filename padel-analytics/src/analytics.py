"""
analytics.py
============
Aggregates all ShotEvent objects produced by the classifier into
meaningful match statistics:

  - Per-player shot counts (forehand / backhand / smash / total)
  - Shot rate  (shots per minute)
  - Rally detection  (bursts of shots within a time window)
  - Shot timeline    (events ordered by time)
  - Court heatmap    (player positions as 2-D density grid)
  - Match summary    (single dict ready for JSON export)

Public API
----------
    analytics = Analytics(frame_w=1280, frame_h=720)
    analytics.update(shot_events, tracks, frame_idx, fps)   # every frame
    summary = analytics.match_summary()
    fig     = analytics.plot_shot_counts()
    fig     = analytics.plot_shot_timeline()
    fig     = analytics.plot_heatmap(track_id=1)
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend — safe in scripts
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    import seaborn as sns
    _PLOT_AVAILABLE = True
except ImportError:
    _PLOT_AVAILABLE = False

from shot_classifier import ShotEvent, ShotType
from tracking import Track
from utils import get_logger

logger = get_logger(__name__)


# ─── Rally dataclass ──────────────────────────────────────────────────────────

@dataclass
class Rally:
    """
    A burst of shots that form one rally.

    A new rally starts when a shot occurs more than
    RALLY_GAP_SECONDS after the previous shot.
    """
    rally_id:      int
    start_frame:   int
    end_frame:     int
    start_time:    float      # seconds
    end_time:      float      # seconds
    shots:         List[ShotEvent] = field(default_factory=list)

    @property
    def duration_sec(self) -> float:
        return round(self.end_time - self.start_time, 2)

    @property
    def shot_count(self) -> int:
        return len(self.shots)

    @property
    def players_involved(self) -> List[int]:
        return list({e.track_id for e in self.shots})

    def to_dict(self) -> dict:
        return {
            "rally_id":         self.rally_id,
            "start_time_sec":   round(self.start_time, 2),
            "end_time_sec":     round(self.end_time, 2),
            "duration_sec":     self.duration_sec,
            "shot_count":       self.shot_count,
            "players_involved": self.players_involved,
        }


# ─── Analytics ────────────────────────────────────────────────────────────────

class Analytics:
    """
    Stateful analytics engine. Feed it data every frame via .update(),
    then query statistics at any time.

    Parameters
    ----------
    frame_w         : int    Video frame width  (pixels)
    frame_h         : int    Video frame height (pixels)
    rally_gap_sec   : float  Silence gap (seconds) that splits two rallies
    heatmap_bins    : int    Grid resolution for court heatmaps
    """

    SHOT_COLOURS = {
        ShotType.FOREHAND: "#2196F3",   # blue
        ShotType.BACKHAND: "#F44336",   # red
        ShotType.SMASH:    "#FF9800",   # orange
        ShotType.UNKNOWN:  "#9E9E9E",   # grey
    }

    def __init__(
        self,
        frame_w:       int   = 1280,
        frame_h:       int   = 720,
        rally_gap_sec: float = 3.0,
        heatmap_bins:  int   = 20,
    ):
        self.frame_w       = frame_w
        self.frame_h       = frame_h
        self.rally_gap_sec = rally_gap_sec
        self.heatmap_bins  = heatmap_bins

        # Shot events accumulated over the whole video
        self._all_events: List[ShotEvent] = []

        # Player position history {track_id: [(cx, cy), ...]}
        self._position_history: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

        # Computed rallies (rebuilt on demand)
        self._rallies: Optional[List[Rally]] = None
        self._rallies_dirty: bool = True

        # Per-player counters for fast access
        self._shot_counts: Dict[int, Dict[str, int]] = defaultdict(
            lambda: {t: 0 for t in ShotType.ALL + ["total"]}
        )

        # Total frames processed (for duration tracking)
        self._total_frames: int = 0
        self._fps: float = 30.0

    # ── Public: update ────────────────────────────────────────────────────────

    def update(
        self,
        new_events: List[ShotEvent],
        tracks:     List[Track],
        frame_idx:  int,
        fps:        float = 30.0,
    ) -> None:
        """
        Feed new data for the current frame.

        Parameters
        ----------
        new_events : shot events emitted this frame (may be empty)
        tracks     : all active tracks this frame (for position history)
        frame_idx  : current frame number
        fps        : video frame rate
        """
        self._fps          = fps
        self._total_frames = max(self._total_frames, frame_idx)

        # ── Accumulate shot events ─────────────────────────────────────────
        for event in new_events:
            self._all_events.append(event)
            self._shot_counts[event.track_id][event.shot_type] += 1
            self._shot_counts[event.track_id]["total"] += 1
            self._rallies_dirty = True

        # ── Record player positions ────────────────────────────────────────
        for track in tracks:
            self._position_history[track.track_id].append(track.center)

    # ── Public: shot counts ───────────────────────────────────────────────────

    def shot_counts(self, track_id: Optional[int] = None
                    ) -> Dict[int, Dict[str, int]]:
        """
        Return shot counts.
        If track_id given, return counts for that player only.
        """
        if track_id is not None:
            return {track_id: dict(self._shot_counts.get(track_id, {}))}
        return {tid: dict(counts)
                for tid, counts in self._shot_counts.items()}

    def total_shots(self) -> int:
        """Total shot events across all players."""
        return len(self._all_events)

    # ── Public: rally detection ────────────────────────────────────────────────

    def rallies(self) -> List[Rally]:
        """
        Return all detected rallies.
        Rallies are computed lazily and cached until new events arrive.
        """
        if self._rallies_dirty or self._rallies is None:
            self._rallies = self._detect_rallies()
            self._rallies_dirty = False
        return self._rallies

    def _detect_rallies(self) -> List[Rally]:
        """
        Split the shot timeline into rallies based on time gaps.
        A new rally begins when two consecutive shots are more than
        self.rally_gap_sec apart.
        """
        if not self._all_events:
            return []

        sorted_events = sorted(self._all_events, key=lambda e: e.timestamp_sec)
        rallies: List[Rally] = []
        rally_id = 1

        current_shots = [sorted_events[0]]

        for event in sorted_events[1:]:
            gap = event.timestamp_sec - current_shots[-1].timestamp_sec
            if gap > self.rally_gap_sec:
                # Close current rally
                rallies.append(self._build_rally(rally_id, current_shots))
                rally_id += 1
                current_shots = [event]
            else:
                current_shots.append(event)

        # Close the last rally
        if current_shots:
            rallies.append(self._build_rally(rally_id, current_shots))

        logger.info(f"Detected {len(rallies)} rallies.")
        return rallies

    @staticmethod
    def _build_rally(rally_id: int, shots: List[ShotEvent]) -> Rally:
        return Rally(
            rally_id=rally_id,
            start_frame=shots[0].frame_idx,
            end_frame=shots[-1].frame_idx,
            start_time=shots[0].timestamp_sec,
            end_time=shots[-1].timestamp_sec,
            shots=list(shots),
        )

    # ── Public: shot rate ─────────────────────────────────────────────────────

    def shot_rate_per_minute(self, track_id: Optional[int] = None) -> float:
        """
        Shots per minute for a player (or all players combined).
        """
        duration_min = (self._total_frames / self._fps) / 60.0
        if duration_min == 0:
            return 0.0
        if track_id is not None:
            count = self._shot_counts.get(track_id, {}).get("total", 0)
        else:
            count = self.total_shots()
        return round(count / duration_min, 2)

    # ── Public: match summary ─────────────────────────────────────────────────

    def match_summary(self) -> dict:
        """
        Return a single dictionary summarising the entire match.
        This is written to shots.json by exporter.py.
        """
        duration_sec = self._total_frames / self._fps if self._fps else 0
        rallies      = self.rallies()

        player_stats = {}
        for tid, counts in self._shot_counts.items():
            player_stats[f"player_{tid}"] = {
                "shot_counts":          dict(counts),
                "shot_rate_per_min":    self.shot_rate_per_minute(tid),
            }

        return {
            "duration_sec":      round(duration_sec, 2),
            "total_frames":      self._total_frames,
            "fps":               self._fps,
            "total_shots":       self.total_shots(),
            "total_rallies":     len(rallies),
            "avg_rally_shots":   round(
                sum(r.shot_count for r in rallies) / len(rallies), 2
            ) if rallies else 0,
            "avg_rally_duration": round(
                sum(r.duration_sec for r in rallies) / len(rallies), 2
            ) if rallies else 0,
            "player_stats":      player_stats,
            "rallies":           [r.to_dict() for r in rallies],
        }

    # ── Public: shot timeline ─────────────────────────────────────────────────

    def shot_timeline(self) -> List[dict]:
        """
        Return all shot events sorted by time as a list of dicts.
        Used for CSV export and timeline plot.
        """
        return [
            e.to_dict()
            for e in sorted(self._all_events, key=lambda e: e.timestamp_sec)
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Plotting helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _check_plot(self) -> bool:
        if not _PLOT_AVAILABLE:
            logger.warning(
                "matplotlib/seaborn not installed. "
                "Run: pip install matplotlib seaborn"
            )
            return False
        return True

    # ── Plot 1: Shot counts bar chart ─────────────────────────────────────────

    def plot_shot_counts(self,
                         save_path: Optional[str] = None):
        """
        Grouped bar chart — forehand / backhand / smash counts per player.
        Returns a matplotlib Figure.
        """
        if not self._check_plot():
            return None

        counts   = self.shot_counts()
        players  = sorted(counts.keys())
        n        = len(players)

        if n == 0:
            logger.warning("No shot data to plot.")
            return None

        shot_types = [ShotType.FOREHAND, ShotType.BACKHAND, ShotType.SMASH]
        x          = np.arange(n)
        width      = 0.25

        fig, ax = plt.subplots(figsize=(max(7, n * 2), 5))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        for i, stype in enumerate(shot_types):
            values = [counts[pid].get(stype, 0) for pid in players]
            bars   = ax.bar(
                x + i * width,
                values,
                width,
                label=stype.capitalize(),
                color=self.SHOT_COLOURS[stype],
                edgecolor="#ffffff22",
                linewidth=0.5,
            )
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.2,
                        str(val),
                        ha="center", va="bottom",
                        fontsize=9, color="white",
                    )

        ax.set_xticks(x + width)
        ax.set_xticklabels(
            [f"Player {pid}" for pid in players],
            color="white", fontsize=11,
        )
        ax.set_ylabel("Shot count", color="white")
        ax.set_title("Shot Distribution per Player", color="white",
                     fontsize=14, fontweight="bold", pad=14)
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#ffffff33")
        ax.yaxis.label.set_color("white")
        ax.legend(facecolor="#0f3460", labelcolor="white", fontsize=10)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            logger.info(f"Shot count chart saved → {save_path}")

        return fig

    # ── Plot 2: Shot timeline ─────────────────────────────────────────────────

    def plot_shot_timeline(self,
                           save_path: Optional[str] = None):
        """
        Scatter plot showing each shot event on a time axis,
        coloured by shot type, shaped by player.
        Returns a matplotlib Figure.
        """
        if not self._check_plot():
            return None

        events = self._all_events
        if not events:
            logger.warning("No events for timeline plot.")
            return None

        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        markers = ["o", "s", "^", "D", "v", "P"]    # one per player

        players = sorted({e.track_id for e in events})
        for event in events:
            pidx   = players.index(event.track_id) % len(markers)
            colour = self.SHOT_COLOURS.get(event.shot_type, "#9E9E9E")
            ax.scatter(
                event.timestamp_sec,
                event.track_id,
                c=colour,
                marker=markers[pidx],
                s=80,
                edgecolors="#ffffff44",
                linewidths=0.5,
                zorder=3,
            )

        # Rally shading
        for rally in self.rallies():
            ax.axvspan(rally.start_time, rally.end_time,
                       alpha=0.08, color="white", zorder=1)

        ax.set_xlabel("Time (seconds)", color="white")
        ax.set_ylabel("Player ID", color="white")
        ax.set_title("Shot Timeline", color="white",
                     fontsize=14, fontweight="bold", pad=14)
        ax.set_yticks(players)
        ax.set_yticklabels([f"Player {p}" for p in players], color="white")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#ffffff33")

        # Legend for shot types
        patches = [
            mpatches.Patch(color=c, label=t.capitalize())
            for t, c in self.SHOT_COLOURS.items()
            if t != ShotType.UNKNOWN
        ]
        ax.legend(handles=patches, facecolor="#0f3460",
                  labelcolor="white", fontsize=9,
                  loc="upper right")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            logger.info(f"Timeline chart saved → {save_path}")

        return fig

    # ── Plot 3: Court heatmap ─────────────────────────────────────────────────

    def plot_heatmap(
        self,
        track_id:  Optional[int] = None,
        save_path: Optional[str] = None,
    ):
        """
        2-D density heatmap of player position(s) on the court.

        Parameters
        ----------
        track_id  : if given, show only that player's heatmap.
                    If None, show all players combined.
        save_path : optional path to save the figure.

        Returns a matplotlib Figure.
        """
        if not self._check_plot():
            return None

        if track_id is not None:
            positions = self._position_history.get(track_id, [])
            title     = f"Player {track_id} — Court Heatmap"
        else:
            positions = [
                pos
                for pid, pos_list in self._position_history.items()
                for pos in pos_list
            ]
            title = "All Players — Court Heatmap"

        if not positions:
            logger.warning("No position data for heatmap.")
            return None

        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        # 2-D histogram heatmap
        h, xedges, yedges = np.histogram2d(
            xs, ys,
            bins=self.heatmap_bins,
            range=[[0, self.frame_w], [0, self.frame_h]],
        )
        # Smooth slightly
        from scipy.ndimage import gaussian_filter
        h_smooth = gaussian_filter(h.T, sigma=1.2)

        im = ax.imshow(
            h_smooth,
            origin="upper",
            extent=[0, self.frame_w, self.frame_h, 0],
            cmap="plasma",
            aspect="auto",
            interpolation="bilinear",
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Density", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        ax.set_xlabel("X (pixels)", color="white")
        ax.set_ylabel("Y (pixels)", color="white")
        ax.set_title(title, color="white",
                     fontsize=13, fontweight="bold", pad=12)
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#ffffff33")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            logger.info(f"Heatmap saved → {save_path}")

        return fig

    # ── Plot 4: Dashboard (all charts combined) ────────────────────────────────

    def plot_dashboard(self, save_path: Optional[str] = None):
        """
        Single figure with three panels:
            Top-left  : shot count bar chart
            Top-right : court heatmap (all players)
            Bottom    : shot timeline

        Returns a matplotlib Figure.
        """
        if not self._check_plot():
            return None

        fig = plt.figure(figsize=(16, 9))
        fig.patch.set_facecolor("#1a1a2e")
        gs  = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

        # ── Top-left: shot counts ──────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        self._draw_shot_counts_on_ax(ax1)

        # ── Top-right: heatmap ────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        self._draw_heatmap_on_ax(ax2)

        # ── Bottom: timeline ──────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[1, :])
        self._draw_timeline_on_ax(ax3)

        fig.suptitle(
            "Padel Match Analytics Dashboard",
            color="white", fontsize=16, fontweight="bold", y=0.98,
        )

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            logger.info(f"Dashboard saved → {save_path}")

        return fig

    # ── Internal draw helpers (reuse logic on arbitrary axes) ─────────────────

    def _draw_shot_counts_on_ax(self, ax) -> None:
        counts    = self.shot_counts()
        players   = sorted(counts.keys())
        n         = len(players)
        if n == 0:
            return
        shot_types = [ShotType.FOREHAND, ShotType.BACKHAND, ShotType.SMASH]
        x          = np.arange(n)
        width      = 0.25
        ax.set_facecolor("#16213e")
        for i, stype in enumerate(shot_types):
            values = [counts[pid].get(stype, 0) for pid in players]
            ax.bar(x + i * width, values, width,
                   label=stype.capitalize(),
                   color=self.SHOT_COLOURS[stype],
                   edgecolor="#ffffff22")
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"P{p}" for p in players], color="white")
        ax.set_title("Shot Counts", color="white", fontsize=11)
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#ffffff33")
        ax.set_facecolor("#16213e")
        ax.legend(facecolor="#0f3460", labelcolor="white", fontsize=8)

    def _draw_heatmap_on_ax(self, ax) -> None:
        positions = [
            pos
            for pos_list in self._position_history.values()
            for pos in pos_list
        ]
        if not positions:
            return
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        h, _, _ = np.histogram2d(
            xs, ys, bins=self.heatmap_bins,
            range=[[0, self.frame_w], [0, self.frame_h]],
        )
        from scipy.ndimage import gaussian_filter
        h_smooth = gaussian_filter(h.T, sigma=1.2)
        ax.imshow(h_smooth, origin="upper",
                  extent=[0, self.frame_w, self.frame_h, 0],
                  cmap="plasma", aspect="auto")
        ax.set_title("Position Heatmap", color="white", fontsize=11)
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#ffffff33")
        ax.set_facecolor("#16213e")

    def _draw_timeline_on_ax(self, ax) -> None:
        events  = self._all_events
        players = sorted({e.track_id for e in events})
        markers = ["o", "s", "^", "D"]
        ax.set_facecolor("#16213e")
        for event in events:
            pidx   = players.index(event.track_id) % len(markers)
            colour = self.SHOT_COLOURS.get(event.shot_type, "#9E9E9E")
            ax.scatter(event.timestamp_sec, event.track_id,
                       c=colour, marker=markers[pidx],
                       s=60, edgecolors="#ffffff44", linewidths=0.5, zorder=3)
        for rally in self.rallies():
            ax.axvspan(rally.start_time, rally.end_time,
                       alpha=0.08, color="white", zorder=1)
        ax.set_xlabel("Time (seconds)", color="white")
        ax.set_ylabel("Player", color="white")
        ax.set_title("Shot Timeline", color="white", fontsize=11)
        ax.set_yticks(players)
        ax.set_yticklabels([f"P{p}" for p in players], color="white")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#ffffff33")

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Analytics("
            f"shots={self.total_shots()}, "
            f"rallies={len(self.rallies())}, "
            f"players={len(self._shot_counts)})"
        )