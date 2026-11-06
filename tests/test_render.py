from PIL import Image

from roadguard.inference.render import draw_detections
from roadguard.inference.schema import Detection


def test_draw_detections_preserves_size_and_does_not_mutate_input():
    image = Image.new("RGB", (200, 100), "white")
    detection = Detection(
        detection_id=1,
        class_id=2,
        class_code="D20",
        damage_type="Alligator Crack",
        confidence=0.9,
        x1=20,
        y1=20,
        x2=100,
        y2=70,
    )
    result = draw_detections(image, [detection])
    assert result.size == image.size
    assert image.getpixel((20, 20)) == (255, 255, 255)
    assert result.getpixel((20, 20)) != (255, 255, 255)
