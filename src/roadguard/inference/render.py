from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from roadguard.constants import CLASS_COLORS_RGB
from roadguard.inference.schema import Detection


def draw_detections(image: Image.Image, detections: list[Detection]) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    stroke = max(2, round(min(canvas.size) / 250))
    for detection in detections:
        color = CLASS_COLORS_RGB[detection.class_id]
        box = (detection.x1, detection.y1, detection.x2, detection.y2)
        draw.rectangle(box, outline=color, width=stroke)
        label = f"{detection.class_code} {detection.confidence:.0%}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        label_width, label_height = right - left + 12, bottom - top + 8
        label_y = max(0, detection.y1 - label_height)
        draw.rounded_rectangle(
            (detection.x1, label_y, detection.x1 + label_width, label_y + label_height),
            radius=4,
            fill=color,
        )
        draw.text((detection.x1 + 6, label_y + 3), label, fill=(7, 18, 38), font=font)
    return canvas
