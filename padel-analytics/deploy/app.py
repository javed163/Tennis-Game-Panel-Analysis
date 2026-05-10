"""
deploy/app.py
=============
FastAPI REST API that wraps the Padel Analytics pipeline.
Upload a video, get back shot events as JSON.

Endpoints
---------
POST /analyze          Upload a video file → runs full pipeline → returns JSON
GET  /results/{job_id} Poll job status and download results
GET  /health           Health check
GET  /                 API docs redirect

Run locally
-----------
    cd deploy/
    pip install fastapi uvicorn python-multipart
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Deploy on Render / Railway / Hugging Face Spaces
-------------------------------------------------
    See deploy/README_DEPLOY.md
"""

from __future__ import annotations

import sys
import uuid
import shutil
import asyncio
from pathlib import Path
from typing import Optional

# ── Add src/ to path ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import OUTPUTS_DIR
from utils  import get_logger, validate_video, get_video_info

logger = get_logger("deploy.app")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Padel Game Analytics API",
    description=(
        "Upload a padel match video and get back "
        "shot classification results as JSON."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job store (replace with Redis for production) ───────────────────
# Structure: { job_id: { "status": str, "result": dict|None, "error": str|None } }
_JOBS: dict = {}

UPLOAD_DIR = ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Background pipeline task
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline_task(job_id: str, video_path: Path, max_frames: int) -> None:
    """
    Run the full analytics pipeline in a background thread.
    Updates _JOBS[job_id] with status and results.
    """
    _JOBS[job_id]["status"] = "processing"
    logger.info(f"Job {job_id}: pipeline starting on {video_path.name}")

    try:
        import cv2
        from detection       import Detector
        from tracking        import Tracker
        from pose_estimator  import PoseEstimator
        from shot_classifier import ShotClassifier
        from analytics       import Analytics
        from visualizer      import Visualizer
        from exporter        import Exporter

        info    = get_video_info(video_path)
        fps     = info["fps"] or 30.0
        frame_w = info["width"]
        frame_h = info["height"]
        total   = info["frame_count"]
        limit   = min(max_frames, total) if max_frames > 0 else total

        out_dir = OUTPUTS_DIR / job_id
        out_dir.mkdir(parents=True, exist_ok=True)

        detector   = Detector(conf=0.4, device="cpu")
        tracker    = Tracker()
        classifier = ShotClassifier()
        analytics  = Analytics(frame_w=frame_w, frame_h=frame_h)
        exporter   = Exporter(output_dir=out_dir)

        try:
            pose_estimator = PoseEstimator()
        except Exception:
            pose_estimator = None
            logger.warning(f"Job {job_id}: pose estimator unavailable.")

        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0

        while cap.isOpened() and frame_idx < limit:
            ret, frame = cap.read()
            if not ret:
                break

            det_result = detector.detect(frame, frame_idx=frame_idx)
            tracks     = tracker.update(det_result, frame)
            poses      = []

            if pose_estimator and tracks:
                poses = pose_estimator.estimate(frame, tracks)

            if poses:
                classifier.feed_batch(poses, frame_idx=frame_idx, fps=fps)

            new_events = classifier.get_new_events()
            analytics.update(new_events=new_events, tracks=tracks,
                             frame_idx=frame_idx, fps=fps)
            frame_idx += 1

        cap.release()
        detector.close()
        if pose_estimator:
            pose_estimator.close()

        all_events    = classifier.all_events
        match_summary = analytics.match_summary()

        exporter.export_json(all_events, match_summary,
                             path=out_dir / "shots.json")
        exporter.export_csv(all_events, path=out_dir / "shots.csv")

        _JOBS[job_id]["status"] = "done"
        _JOBS[job_id]["result"] = {
            "job_id":        job_id,
            "total_shots":   len(all_events),
            "total_rallies": len(analytics.rallies()),
            "duration_sec":  match_summary.get("duration_sec", 0),
            "shots":         [e.to_dict() for e in all_events],
            "summary":       match_summary,
            "files": {
                "json": f"/results/{job_id}/shots.json",
                "csv":  f"/results/{job_id}/shots.csv",
            },
        }
        logger.info(f"Job {job_id}: done. Shots={len(all_events)}")

    except Exception as exc:
        _JOBS[job_id]["status"] = "error"
        _JOBS[job_id]["error"]  = str(exc)
        logger.error(f"Job {job_id}: FAILED — {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    """Redirect root to interactive API docs."""
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "Padel Analytics API"}


@app.post("/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    file:             UploadFile = File(..., description="Padel match video (.mp4)"),
    max_frames:       int        = 300,
):
    """
    Upload a padel match video and start analysis.

    - **file**: MP4 video file
    - **max_frames**: max frames to process (0 = all). Default 300 for speed.

    Returns a **job_id** — poll `/results/{job_id}` for status and results.
    """
    # ── Validate file type ────────────────────────────────────────────────
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(
            status_code=400,
            detail="Only video files are accepted (.mp4, .avi, .mov, .mkv)",
        )

    # ── Save uploaded file ────────────────────────────────────────────────
    job_id     = str(uuid.uuid4())[:8]
    save_path  = UPLOAD_DIR / f"{job_id}_{file.filename}"

    with open(save_path, "wb") as f_out:
        shutil.copyfileobj(file.file, f_out)

    # ── Validate video ────────────────────────────────────────────────────
    ok, msg = validate_video(save_path)
    if not ok:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Invalid video: {msg}")

    # ── Register job ──────────────────────────────────────────────────────
    _JOBS[job_id] = {"status": "queued", "result": None, "error": None}

    # ── Queue background task ─────────────────────────────────────────────
    background_tasks.add_task(
        _run_pipeline_task, job_id, save_path, max_frames
    )

    logger.info(f"Job {job_id} queued: {file.filename}")

    return JSONResponse({
        "job_id":       job_id,
        "status":       "queued",
        "poll_url":     f"/results/{job_id}",
        "message":      "Video received. Poll /results/{job_id} for status.",
    })


@app.get("/results/{job_id}")
def get_results(job_id: str):
    """
    Poll job status.

    Returns:
    - `status: queued`     — not started yet
    - `status: processing` — pipeline running
    - `status: done`       — results ready (includes full shot list)
    - `status: error`      — pipeline failed (includes error message)
    """
    if job_id not in _JOBS:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    job = _JOBS[job_id]

    if job["status"] == "done":
        return JSONResponse(job["result"])

    if job["status"] == "error":
        raise HTTPException(status_code=500,
                            detail=f"Pipeline error: {job['error']}")

    return JSONResponse({"job_id": job_id, "status": job["status"]})


@app.get("/results/{job_id}/shots.json")
def download_json(job_id: str):
    """Download the shots.json file for a completed job."""
    path = OUTPUTS_DIR / job_id / "shots.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not ready yet.")
    return FileResponse(str(path), media_type="application/json",
                        filename="shots.json")


@app.get("/results/{job_id}/shots.csv")
def download_csv(job_id: str):
    """Download the shots.csv file for a completed job."""
    path = OUTPUTS_DIR / job_id / "shots.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not ready yet.")
    return FileResponse(str(path), media_type="text/csv",
                        filename="shots.csv")


@app.get("/jobs")
def list_jobs():
    """List all jobs and their statuses."""
    return JSONResponse({
        jid: {"status": info["status"]}
        for jid, info in _JOBS.items()
    })


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)