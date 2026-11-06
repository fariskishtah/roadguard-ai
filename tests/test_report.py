from PIL import Image

from roadguard.inference.report import create_pdf_report
from roadguard.inference.schema import Prediction


def test_pdf_report_is_generated_for_empty_detection(tmp_path):
    prediction = Prediction(
        filename="clear-road.jpg",
        model_version="test.pt",
        image_width=320,
        image_height=240,
        inference_ms=5.0,
        confidence_threshold=0.25,
        nms_iou_threshold=0.45,
        detections=[],
    )
    output = create_pdf_report(
        prediction, Image.new("RGB", (320, 240), "gray"), tmp_path / "report.pdf"
    )
    assert output.read_bytes().startswith(b"%PDF")
    assert output.stat().st_size > 1_000
