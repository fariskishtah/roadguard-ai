import csv
import json

import pytest

from roadguard.inference.export import export_csv, export_json, prediction_rows
from roadguard.inference.schema import Detection, Prediction


def sample_prediction() -> Prediction:
    return Prediction(
        filename="road.jpg",
        model_version="best.pt",
        image_width=640,
        image_height=480,
        inference_ms=12.5,
        confidence_threshold=0.25,
        nms_iou_threshold=0.45,
        detections=[
            Detection(
                detection_id=1,
                class_id=3,
                class_code="D40",
                damage_type="Pothole",
                confidence=0.87,
                x1=10,
                y1=20,
                x2=110,
                y2=80,
            )
        ],
    )


def test_schema_rejects_inconsistent_class():
    with pytest.raises(ValueError, match="class_code"):
        Detection(
            detection_id=1,
            class_id=0,
            class_code="D40",
            damage_type="Longitudinal Crack",
            confidence=0.5,
            x1=0,
            y1=0,
            x2=1,
            y2=1,
        )


def test_exports_are_machine_readable(tmp_path):
    prediction = sample_prediction()
    assert prediction_rows(prediction)[0]["width"] == 100
    json_path = export_json(prediction, tmp_path / "result.json")
    csv_path = export_csv(prediction, tmp_path / "result.csv")
    assert json.loads(json_path.read_text())["detections"][0]["class_code"] == "D40"
    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["damage_type"] == "Pothole"


def test_empty_detection_export_keeps_csv_header(tmp_path):
    prediction = sample_prediction().model_copy(update={"detections": []})
    csv_path = export_csv(prediction, tmp_path / "empty.csv")
    assert len(csv_path.read_text().splitlines()) == 1
