from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from roadguard.constants import CLASS_CODES, CLASS_NAMES
from roadguard.inference.render import draw_detections
from roadguard.inference.schema import Detection, Prediction


class RoadDamageDetector:
    def __init__(self, weights: str | Path):
        from ultralytics import YOLO

        self.weights = Path(weights)
        if not self.weights.is_file():
            raise FileNotFoundError(f"Production checkpoint not found: {self.weights}")
        self.model = YOLO(str(self.weights))
        names = self.model.names
        ordered = [names[index] for index in range(len(names))]
        accepted = [list(CLASS_NAMES), list(CLASS_CODES)]
        if ordered not in accepted:
            raise ValueError(f"Checkpoint classes are incompatible: {ordered}")

    def predict(
        self,
        source: Image.Image | str | Path,
        *,
        confidence: float = 0.25,
        nms_iou: float = 0.45,
        image_size: int = 640,
        filename: str | None = None,
    ) -> tuple[Prediction, Image.Image]:
        if not 0 <= confidence <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        if not 0 < nms_iou <= 1:
            raise ValueError("NMS IoU threshold must be in (0, 1]")
        if isinstance(source, Image.Image):
            image = ImageOps.exif_transpose(source).convert("RGB")
            input_name = filename or "uploaded-image"
        else:
            path = Path(source)
            input_name = filename or path.name
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
        start = time.perf_counter()
        results = self.model.predict(
            source=image,
            conf=confidence,
            iou=nms_iou,
            imgsz=image_size,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        detections: list[Detection] = []
        boxes: Any = results[0].boxes
        if boxes is not None:
            for index, (xyxy, class_id, score) in enumerate(
                zip(
                    boxes.xyxy.cpu().tolist(),
                    boxes.cls.cpu().tolist(),
                    boxes.conf.cpu().tolist(),
                    strict=True,
                ),
                start=1,
            ):
                cid = int(class_id)
                detections.append(
                    Detection(
                        detection_id=index,
                        class_id=cid,
                        class_code=CLASS_CODES[cid],
                        damage_type=CLASS_NAMES[cid],
                        confidence=float(score),
                        x1=max(0.0, min(float(image.width), float(xyxy[0]))),
                        y1=max(0.0, min(float(image.height), float(xyxy[1]))),
                        x2=max(0.0, min(float(image.width), float(xyxy[2]))),
                        y2=max(0.0, min(float(image.height), float(xyxy[3]))),
                    )
                )
        prediction = Prediction(
            filename=input_name,
            model_version=self.weights.name,
            image_width=image.width,
            image_height=image.height,
            inference_ms=elapsed_ms,
            confidence_threshold=confidence,
            nms_iou_threshold=nms_iou,
            detections=detections,
        )
        return prediction, draw_detections(image, detections)
