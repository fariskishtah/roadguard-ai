from __future__ import annotations

import json
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from roadguard.inference.export import prediction_rows

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}


def process_batch(
    detector,
    image_paths: list[str | Path],
    *,
    confidence: float,
    nms_iou: float,
    max_images: int = 20,
) -> tuple[Path, pd.DataFrame]:
    if not image_paths:
        raise ValueError("Select at least one image")
    if len(image_paths) > max_images:
        raise ValueError(f"Batch limit is {max_images} images")
    work = Path(tempfile.mkdtemp(prefix="roadguard-batch-"))
    rows: list[dict] = []
    summaries = []
    for index, raw_path in enumerate(image_paths, start=1):
        path = Path(raw_path)
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image format: {path.suffix}")
        prediction, annotated = detector.predict(
            path,
            confidence=confidence,
            nms_iou=nms_iou,
            filename=path.name,
        )
        annotated_name = f"{index:03d}_{path.stem}_annotated.jpg"
        annotated.save(work / annotated_name, quality=92)
        image_rows = prediction_rows(prediction)
        rows.extend(image_rows)
        summaries.append(
            {
                "filename": path.name,
                "detections": len(prediction.detections),
                "inference_ms": prediction.inference_ms,
                "counts": dict(Counter(det.damage_type for det in prediction.detections)),
            }
        )
        (work / f"{index:03d}_{path.stem}.json").write_text(
            json.dumps(prediction.model_dump(), indent=2) + "\n", encoding="utf-8"
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(work / "detections.csv", index=False)
    (work / "batch_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    archive = work / "roadguard-batch-results.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in work.iterdir():
            if path != archive:
                bundle.write(path, arcname=path.name)
    return archive, pd.DataFrame(summaries)


def process_video(
    detector,
    video_path: str | Path,
    *,
    confidence: float,
    nms_iou: float,
    max_duration_seconds: float = 90.0,
    max_width: int = 1280,
) -> tuple[Path, dict]:
    video_path = Path(video_path)
    if video_path.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError(f"Unsupported video format: {video_path.suffix}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("The uploaded video could not be opened")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError("The video metadata is invalid")
    duration = frame_count / fps
    if duration > max_duration_seconds:
        capture.release()
        raise ValueError(f"Video limit is {max_duration_seconds:.0f} seconds")
    if width > max_width:
        scale = max_width / width
        output_width = max_width
        output_height = max(2, int(height * scale) // 2 * 2)
    else:
        output_width, output_height = width, height
    work = Path(tempfile.mkdtemp(prefix="roadguard-video-"))
    output_path = work / "annotated-video.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (output_width, output_height),
    )
    processed = 0
    detections = 0
    started = time.perf_counter()
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            if (bgr.shape[1], bgr.shape[0]) != (output_width, output_height):
                bgr = cv2.resize(bgr, (output_width, output_height), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            prediction, annotated = detector.predict(
                Image.fromarray(rgb),
                confidence=confidence,
                nms_iou=nms_iou,
                filename=f"frame-{processed:06d}",
            )
            detections += len(prediction.detections)
            writer.write(cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR))
            processed += 1
    finally:
        capture.release()
        writer.release()
    elapsed = time.perf_counter() - started
    if processed == 0 or not output_path.exists():
        raise ValueError("No video frames were decoded")
    summary = {
        "frames": processed,
        "source_fps": fps,
        "processing_fps": processed / elapsed,
        "elapsed_seconds": elapsed,
        "detections_across_frames": detections,
        "output_resolution": f"{output_width}x{output_height}",
    }
    (work / "video_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return output_path, summary
