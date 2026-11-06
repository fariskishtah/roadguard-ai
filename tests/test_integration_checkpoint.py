from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

from roadguard.constants import CLASS_NAMES
from roadguard.inference.engine import RoadDamageDetector


@pytest.mark.integration
def test_real_checkpoint_loads_and_returns_valid_prediction() -> None:
    configured = os.environ.get("ROADGUARD_TEST_WEIGHTS")
    if not configured:
        pytest.skip("Set ROADGUARD_TEST_WEIGHTS to run the real-checkpoint integration test")
    weights = Path(configured)
    assert weights.is_file(), f"Checkpoint does not exist: {weights}"
    detector = RoadDamageDetector(weights)
    assert [detector.model.names[index] for index in range(4)] == list(CLASS_NAMES)
    prediction, annotated = detector.predict(Image.new("RGB", (640, 480), "#707070"))
    assert prediction.image_width == 640
    assert prediction.image_height == 480
    assert prediction.model_version == weights.name
    assert prediction.inference_ms > 0
    assert annotated.size == (640, 480)
    assert all(0 <= detection.class_id < 4 for detection in prediction.detections)
