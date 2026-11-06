from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from roadguard.inference.processing import process_batch, process_video
from roadguard.inference.schema import Prediction


class FakeDetector:
    def predict(self, source, *, confidence, nms_iou, filename):
        if isinstance(source, (str, Path)):
            with Image.open(source) as opened:
                image = opened.convert("RGB")
        else:
            image = source.convert("RGB")
        return (
            Prediction(
                filename=filename,
                model_version="fake-test-checkpoint.pt",
                image_width=image.width,
                image_height=image.height,
                inference_ms=1.0,
                confidence_threshold=confidence,
                nms_iou_threshold=nms_iou,
                detections=[],
            ),
            image,
        )


def test_batch_processing_exports_archive_and_summary(tmp_path):
    first = tmp_path / "one.jpg"
    second = tmp_path / "two.png"
    Image.new("RGB", (32, 24), "gray").save(first)
    Image.new("RGB", (32, 24), "black").save(second)
    archive, summary = process_batch(FakeDetector(), [first, second], confidence=0.25, nms_iou=0.45)
    assert archive.exists()
    assert len(summary) == 2
    assert summary["detections"].sum() == 0


def test_batch_processing_rejects_unsupported_files(tmp_path):
    bad = tmp_path / "file.txt"
    bad.write_text("x")
    with pytest.raises(ValueError, match="Unsupported"):
        process_batch(FakeDetector(), [bad], confidence=0.25, nms_iou=0.45)


def test_video_processing_handles_empty_predictions(tmp_path):
    source = tmp_path / "road.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 5, (32, 24))
    for value in (20, 80, 140):
        writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
    writer.release()
    output, summary = process_video(
        FakeDetector(), source, confidence=0.25, nms_iou=0.45, max_duration_seconds=5
    )
    assert output.exists()
    assert summary["frames"] == 3
    assert summary["detections_across_frames"] == 0
