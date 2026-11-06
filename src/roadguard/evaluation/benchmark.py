from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


def benchmark_model(
    weights: str | Path,
    image_paths: list[str | Path],
    output_path: str | Path,
    *,
    image_size: int = 640,
    warmup: int = 5,
    repetitions: int = 30,
    device: str | int | None = None,
) -> dict:
    if not image_paths:
        raise ValueError("At least one benchmark image is required")
    model = YOLO(str(weights))
    prepared = []
    for path in image_paths:
        with Image.open(path) as image:
            prepared.append(image.convert("RGB").copy())
    for index in range(warmup):
        model.predict(
            prepared[index % len(prepared)], imgsz=image_size, device=device, verbose=False
        )
    model_only: list[float] = []
    end_to_end: list[float] = []
    for index in range(repetitions):
        path = Path(image_paths[index % len(image_paths)])
        start = time.perf_counter()
        with Image.open(path) as image:
            frame = image.convert("RGB")
            before_model = time.perf_counter()
            model.predict(frame, imgsz=image_size, device=device, verbose=False)
            after_model = time.perf_counter()
        end = time.perf_counter()
        model_only.append((after_model - before_model) * 1000)
        end_to_end.append((end - start) * 1000)

    def summarize(values: list[float]) -> dict:
        ordered = sorted(values)
        p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
        median = statistics.median(ordered)
        return {"median_ms": median, "p95_ms": ordered[p95_index], "fps": 1000.0 / median}

    payload = {
        "weights": str(weights),
        "device": str(device),
        "image_size": image_size,
        "batch_size": 1,
        "warmup": warmup,
        "repetitions": repetitions,
        "model_only": summarize(model_only),
        "end_to_end": summarize(end_to_end),
    }
    Path(output_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
