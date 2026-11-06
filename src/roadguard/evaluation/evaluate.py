from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLO

from roadguard.constants import CLASS_CODES, CLASS_NAMES


def evaluate_checkpoint(
    weights: str | Path,
    data_yaml: str | Path,
    output_path: str | Path,
    *,
    split: str = "test",
    image_size: int = 640,
    batch: int = 16,
    device: str | int | None = None,
) -> dict:
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=image_size,
        batch=batch,
        device=device,
        plots=True,
        verbose=False,
    )
    box = metrics.box
    per_class = []
    for class_id, code in enumerate(CLASS_CODES):
        per_class.append(
            {
                "class_id": class_id,
                "class_code": code,
                "class_name": CLASS_NAMES[class_id],
                "ap50": float(box.ap50[class_id]),
                "ap50_95": float(box.ap[class_id]),
                "precision": float(box.p[class_id]),
                "recall": float(box.r[class_id]),
            }
        )
    payload = {
        "weights": str(weights),
        "split": split,
        "image_size": image_size,
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "per_class": per_class,
        "speed_ms": {key: float(value) for key, value in metrics.speed.items()},
        "parameters": sum(parameter.numel() for parameter in model.model.parameters()),
        "size_mb": Path(weights).stat().st_size / (1024 * 1024),
    }
    Path(output_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
