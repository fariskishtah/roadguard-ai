from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from roadguard.constants import CODE_TO_ID


@dataclass(frozen=True)
class VocBox:
    class_code: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass(frozen=True)
class VocAnnotation:
    filename: str
    width: int
    height: int
    boxes: tuple[VocBox, ...]


def parse_voc_xml(path: str | Path) -> VocAnnotation:
    root = ET.parse(path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing <size> in {path}")
    width = int(size.findtext("width", "0"))
    height = int(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions in {path}: {width}x{height}")
    boxes: list[VocBox] = []
    for obj in root.findall("object"):
        code = obj.findtext("name", "").strip()
        box = obj.find("bndbox")
        if box is None:
            continue
        boxes.append(
            VocBox(
                class_code=code,
                xmin=float(box.findtext("xmin", "nan")),
                ymin=float(box.findtext("ymin", "nan")),
                xmax=float(box.findtext("xmax", "nan")),
                ymax=float(box.findtext("ymax", "nan")),
            )
        )
    return VocAnnotation(
        filename=root.findtext("filename", Path(path).with_suffix(".jpg").name),
        width=width,
        height=height,
        boxes=tuple(boxes),
    )


def clip_box(box: VocBox, width: int, height: int) -> VocBox:
    return VocBox(
        box.class_code,
        max(0.0, min(float(width), box.xmin)),
        max(0.0, min(float(height), box.ymin)),
        max(0.0, min(float(width), box.xmax)),
        max(0.0, min(float(height), box.ymax)),
    )


def voc_box_to_yolo(box: VocBox, width: int, height: int) -> tuple[int, float, float, float, float]:
    if box.class_code not in CODE_TO_ID:
        raise ValueError(f"Unsupported RDD2022 class: {box.class_code}")
    clipped = clip_box(box, width, height)
    if clipped.xmax <= clipped.xmin or clipped.ymax <= clipped.ymin:
        raise ValueError(f"Degenerate box: {box}")
    x_center = ((clipped.xmin + clipped.xmax) / 2.0) / width
    y_center = ((clipped.ymin + clipped.ymax) / 2.0) / height
    box_width = (clipped.xmax - clipped.xmin) / width
    box_height = (clipped.ymax - clipped.ymin) / height
    values = (x_center, y_center, box_width, box_height)
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"Invalid normalized box: {values}")
    return (CODE_TO_ID[box.class_code], *values)


def annotation_to_yolo_lines(annotation: VocAnnotation) -> list[str]:
    lines: list[str] = []
    for box in annotation.boxes:
        if box.class_code not in CODE_TO_ID:
            continue
        class_id, x, y, w, h = voc_box_to_yolo(box, annotation.width, annotation.height)
        lines.append(f"{class_id} {x:.8f} {y:.8f} {w:.8f} {h:.8f}")
    return lines
