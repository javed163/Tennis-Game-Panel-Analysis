"""
main.py
=======
Command-line entry point for the Padel Game Analytics pipeline.

Wires together every module in order:
    Video input
        → Detector       (detection.py)
        → Tracker        (tracking.py)
        → PoseEstimator  (pose_estimator.py)
        → ShotClassifier (shot_classifier.py)
        → Analytics      (analytics.py)
        → Visualizer     (visualizer.py)
        → Exporter       (exporter.py)

Usage
-----
    # Basic run
    python main.py --input data/raw/match.mp4

    # Full options
    python main.py \
        --input   data/raw/match.mp4 \
        --output  data/outputs/ \
        --device  cpu \
        --conf    0.4 \
        --no-video \
        --no-skeleton \
        --no-hud \
        --max-frames 500

    # Use GPU (if available)
    python main.py --input data/raw/match.mp4 --device 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
from tqdm import tqdm

# ── Add src/ to path so all module imports work ────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import (
    DEFAULT_INPUT_VIDEO,
    OUTPUTS_DIR,
    YOLO_CONF_THRESHOLD,
    YOLO_DEVICE,
    OUTPUT_FPS,
)
from detection      import Detector
from tracking       import Tracker
from pose_estimator import PoseEstimator
from shot_classifier import ShotClassifier
from analytics      import Analytics
from visualizer     import Visualizer
from exporter       import Exporter
from utils          import get_logger, validate_video, get_video_info

logger = get_logger("main")


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="padel-analytics",
        description="Padel Game Analytics — Shot Classification System",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── I/O ───────────────────────────────────────────────────────────────────
    p.add_argument(
        "--input", "-i",
        type=str,
        default=str(DEFAULT_INPUT_VIDEO),
        metavar="VIDEO",
        help="Path to input padel match video  (default: data/raw/match.mp4)",
    )
    p.add_argument(
        "--output", "-o",
        type=str,
        default=str(OUTPUTS_DIR),
        metavar="DIR",
        help="Output directory for all results  (default: data/outputs/)",
    )

    # ── Detection ─────────────────────────────────────────────────────────────
    p.add_argument(
        "--conf",
        type=float,
        default=YOLO_CONF_THRESHOLD,
        metavar="FLOAT",
        help=f"YOLO confidence threshold  (default: {YOLO_CONF_THRESHOLD})",
    )
    p.add_argument(
        "--device",
        type=str,
        default=YOLO_DEVICE,
        metavar="STR",
        help="Inference device: 'cpu' or '0' for GPU  (default: cpu)",
    )

    # ── Pipeline toggles ──────────────────────────────────────────────────────
    p.add_argument(
        "--no-video",
        action="store_true",
        help="Skip writing the annotated output video (faster)",
    )
    p.add_argument(
        "--no-skeleton",
        action="store_true",
        help="Disable skeleton overlay on output video",
    )
    p.add_argument(
        "--no-hud",
        action="store_true",
        help="Disable HUD panel on output video",
    )
    p.add_argument(
        "--no-pose",
        action="store_true",
        help="Skip pose estimation (faster, disables shot classification)",
    )

    # ── Limits ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        metavar="N",
        help="Stop after N frames  (0 = process entire video)",
    )
    p.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        metavar="N",
        help="Process every Nth frame  (0 or 1 = every frame)",
    )

    # ── Velocity threshold ────────────────────────────────────────────────────
    p.add_argument(
        "--vel-threshold",
        type=float,
        default=8.0,
        metavar="FLOAT",
        help="Wrist velocity threshold to trigger a shot event  (default: 8.0)",
    )

    return p


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(args: argparse.Namespace) -> None:
    """
    Main pipeline loop.

    Opens the video, processes each frame through the full
    detection → tracking → pose → classification → analytics → visualize
    chain, then exports all results.
    """

    input_path  = Path(args.input)
    output_dir  = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Validate input ────────────────────────────────────────────────────────
    ok, msg = validate_video(input_path)
    if not ok:
        logger.error(f"Invalid input video: {msg}")
        sys.exit(1)

    info = get_video_info(input_path)
    fps         = info["fps"] or OUTPUT_FPS
    frame_w     = info["width"]
    frame_h     = info["height"]
    total_frames = info["frame_count"]

    logger.info("=" * 56)
    logger.info("  PADEL ANALYTICS PIPELINE  —  START")
    logger.info("=" * 56)
    logger.info(f"  Input      : {input_path}")
    logger.info(f"  Output dir : {output_dir}")
    logger.info(f"  Resolution : {frame_w}x{frame_h}  @  {fps:.1f} fps")
    logger.info(f"  Frames     : {total_frames}")
    logger.info(f"  Device     : {args.device}")
    logger.info(f"  Conf       : {args.conf}")
    logger.info("=" * 56)

    # ── Initialise all modules ────────────────────────────────────────────────
    logger.info("Initialising modules...")

    detector   = Detector(conf=args.conf, device=args.device)
    tracker    = Tracker()
    classifier = ShotClassifier(velocity_threshold=args.vel_threshold)
    analytics  = Analytics(frame_w=frame_w, frame_h=frame_h)
    visualizer = Visualizer(
        frame_w=frame_w,
        frame_h=frame_h,
        show_skeleton=not args.no_skeleton,
        show_hud=not args.no_hud,
    )
    exporter   = Exporter(output_dir=output_dir)

    # Pose estimator (optional)
    pose_estimator = None
    if not args.no_pose:
        try:
            pose_estimator = PoseEstimator()
        except ImportError:
            logger.warning(
                "MediaPipe not installed — pose estimation disabled. "
                "Run: pip install mediapipe"
            )

    # ── Open video writer ─────────────────────────────────────────────────────
    if not args.no_video:
        video_out = output_dir / "output_annotated.mp4"
        exporter.open_video_writer(frame_w, frame_h, fps, path=video_out)

    # ── Open video capture ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        logger.error("Failed to open video capture.")
        sys.exit(1)

    # ── Determine frame limit ─────────────────────────────────────────────────
    max_frames   = args.max_frames if args.max_frames > 0 else total_frames
    skip         = max(1, args.skip_frames) if args.skip_frames > 1 else 1
    frames_to_process = min(max_frames, total_frames)

    # ── Stats counters ────────────────────────────────────────────────────────
    frame_idx        = 0
    processed_count  = 0
    t_start          = time.perf_counter()

    # ── Main loop ─────────────────────────────────────────────────────────────
    logger.info("Starting frame loop...")

    pbar = tqdm(
        total=frames_to_process,
        desc="Processing",
        unit="frame",
        ncols=80,
        colour="green",
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ── Frame skipping ────────────────────────────────────────────────
        if frame_idx % skip != 0:
            frame_idx += 1
            continue

        # ── Max-frames guard ──────────────────────────────────────────────
        if processed_count >= max_frames:
            break

        # ── 1. Detection ──────────────────────────────────────────────────
        det_result = detector.detect(frame, frame_idx=frame_idx)

        # ── 2. Tracking ───────────────────────────────────────────────────
        tracks = tracker.update(det_result, frame)

        # ── 3. Pose estimation ────────────────────────────────────────────
        poses = []
        if pose_estimator and tracks:
            poses = pose_estimator.estimate(frame, tracks)

        # ── 4. Shot classification ────────────────────────────────────────
        new_events: list = []
        if poses:
            classifier.feed_batch(poses, frame_idx=frame_idx, fps=fps)
            new_events = classifier.get_new_events()

        # ── 5. Analytics update ───────────────────────────────────────────
        analytics.update(
            new_events=new_events,
            tracks=tracks,
            frame_idx=frame_idx,
            fps=fps,
        )

        # ── 6. Visualize ──────────────────────────────────────────────────
        annotated = visualizer.draw(
            frame=frame,
            det_result=det_result,
            tracks=tracks,
            poses=poses,
            new_events=new_events,
            shot_counts=analytics.shot_counts(),
            frame_idx=frame_idx,
            fps=fps,
        )

        # ── 7. Write frame ────────────────────────────────────────────────
        if not args.no_video:
            exporter.write_frame(annotated)

        # ── Progress update ───────────────────────────────────────────────
        pbar.set_postfix({
            "players": det_result.player_count,
            "shots":   analytics.total_shots(),
            "tracks":  tracker.track_count,
        })
        pbar.update(1)

        frame_idx       += 1
        processed_count += 1

    pbar.close()
    cap.release()

    # ── Close video writer ────────────────────────────────────────────────────
    if not args.no_video:
        exporter.close_video_writer()

    # ── Close pose estimator ──────────────────────────────────────────────────
    if pose_estimator:
        pose_estimator.close()

    # ── Close detector ────────────────────────────────────────────────────────
    detector.close()

    # ── Pipeline timing ───────────────────────────────────────────────────────
    elapsed     = time.perf_counter() - t_start
    avg_fps     = processed_count / elapsed if elapsed > 0 else 0

    logger.info("=" * 56)
    logger.info(f"  Frames processed : {processed_count}")
    logger.info(f"  Total shots      : {analytics.total_shots()}")
    logger.info(f"  Rallies detected : {len(analytics.rallies())}")
    logger.info(f"  Elapsed time     : {elapsed:.1f}s")
    logger.info(f"  Avg throughput   : {avg_fps:.1f} fps")
    logger.info("=" * 56)

    # ── Export all results ────────────────────────────────────────────────────
    logger.info("Exporting results...")

    all_events    = classifier.all_events
    match_summary = analytics.match_summary()

    exporter.export_all(
        shot_events=all_events,
        match_summary=match_summary,
        analytics=analytics,
    )

    # ── Final summary to console ──────────────────────────────────────────────
    _print_final_summary(analytics, all_events)


# ──────────────────────────────────────────────────────────────────────────────
# Console summary
# ──────────────────────────────────────────────────────────────────────────────

def _print_final_summary(analytics, all_events) -> None:
    """Print a human-readable match summary to stdout."""
    counts  = analytics.shot_counts()
    rallies = analytics.rallies()

    print()
    print("╔══════════════════════════════════════════════╗")
    print("║        PADEL ANALYTICS — MATCH SUMMARY       ║")
    print("╠══════════════════════════════════════════════╣")
    print(f"║  Total shots   : {analytics.total_shots():<27}║")
    print(f"║  Total rallies : {len(rallies):<27}║")

    if rallies:
        avg_dur = sum(r.duration_sec for r in rallies) / len(rallies)
        print(f"║  Avg rally dur : {avg_dur:<.1f}s{'':<25}║")

    print("╠══════════════════════════════════════════════╣")

    for tid in sorted(counts.keys()):
        c  = counts[tid]
        fh = c.get("forehand", 0)
        bh = c.get("backhand", 0)
        sm = c.get("smash",    0)
        tt = c.get("total",    0)
        rate = analytics.shot_rate_per_minute(tid)
        print(f"║  Player {tid:<3}  Total={tt:<3}  "
              f"FH={fh:<3} BH={bh:<3} SM={sm:<3}     ║")
        print(f"║             Rate={rate:<.1f} shots/min{'':<14}║")

    print("╚══════════════════════════════════════════════╝")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()