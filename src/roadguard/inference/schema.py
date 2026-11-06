from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from roadguard.constants import CLASS_CODES, CLASS_NAMES


class Detection(BaseModel):
    detection_id: int = Field(ge=1)
    class_id: int = Field(ge=0, le=3)
    class_code: str
    damage_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def validate_detection(self) -> Detection:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("Bounding-box maximums must not precede minimums")
        if self.class_code != CLASS_CODES[self.class_id]:
            raise ValueError("class_code does not match class_id")
        if self.damage_type != CLASS_NAMES[self.class_id]:
            raise ValueError("damage_type does not match class_id")
        return self

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


class Prediction(BaseModel):
    filename: str
    model_version: str
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    inference_ms: float = Field(ge=0)
    confidence_threshold: float = Field(ge=0, le=1)
    nms_iou_threshold: float = Field(gt=0, le=1)
    detections: list[Detection]
