from __future__ import annotations

import csv
import json
from pathlib import Path

from roadguard.inference.schema import Prediction


def prediction_rows(prediction: Prediction) -> list[dict]:
    return [
        {
            "filename": prediction.filename,
            "detection_id": det.detection_id,
            "class_id": det.class_id,
            "class_code": det.class_code,
            "damage_type": det.damage_type,
            "confidence": det.confidence,
            "x1": det.x1,
            "y1": det.y1,
            "x2": det.x2,
            "y2": det.y2,
            "width": det.width,
            "height": det.height,
        }
        for det in prediction.detections
    ]


def export_json(prediction: Prediction, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(prediction.model_dump(), indent=2) + "\n", encoding="utf-8")
    return path


def export_csv(prediction: Prediction, path: str | Path) -> Path:
    path = Path(path)
    rows = prediction_rows(prediction)
    fields = [
        "filename",
        "detection_id",
        "class_id",
        "class_code",
        "damage_type",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "width",
        "height",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path
