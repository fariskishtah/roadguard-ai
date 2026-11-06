from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image as PdfImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from roadguard.inference.schema import Prediction


def create_pdf_report(
    prediction: Prediction,
    annotated_image: Image.Image,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    image_path = output_path.with_suffix(".annotated.jpg")
    annotated_image.convert("RGB").save(image_path, quality=90)
    styles = getSampleStyleSheet()
    document = SimpleDocTemplate(
        str(output_path), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm
    )
    elements = [
        Paragraph("RoadGuard AI — Road Inspection Report", styles["Title"]),
        Spacer(1, 6 * mm),
        Table(
            [
                ["Input", prediction.filename],
                ["Processed", datetime.now().astimezone().isoformat(timespec="seconds")],
                ["Model", prediction.model_version],
                ["Detections", str(len(prediction.detections))],
                ["Inference", f"{prediction.inference_ms:.1f} ms"],
                ["Confidence threshold", f"{prediction.confidence_threshold:.2f}"],
            ],
            colWidths=[45 * mm, 115 * mm],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#0B1F3A")),
                    ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DCE6F2")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("PADDING", (0, 0), (-1, -1), 7),
                ]
            ),
        ),
        Spacer(1, 6 * mm),
        PdfImage(str(image_path), width=160 * mm, height=95 * mm, kind="proportional"),
        Spacer(1, 5 * mm),
    ]
    rows = [["ID", "Code", "Damage type", "Confidence", "Bounding box"]]
    for detection in prediction.detections:
        rows.append(
            [
                str(detection.detection_id),
                detection.class_code,
                detection.damage_type,
                f"{detection.confidence:.1%}",
                (
                    f"({detection.x1:.0f}, {detection.y1:.0f})–"
                    f"({detection.x2:.0f}, {detection.y2:.0f})"
                ),
            ]
        )
    if len(rows) == 1:
        rows.append(["—", "—", "No detections above threshold", "—", "—"])
    elements.append(
        Table(
            rows,
            repeatRows=1,
            colWidths=[12 * mm, 18 * mm, 45 * mm, 27 * mm, 58 * mm],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1261A0")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DCE6F2")),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            ),
        )
    )
    elements.extend(
        [
            Spacer(1, 5 * mm),
            Paragraph(
                "This report records model detections only. It does not establish "
                "physical severity, "
                "repair cost, or road safety, and must be reviewed by a qualified inspector.",
                styles["BodyText"],
            ),
        ]
    )
    document.build(elements)
    image_path.unlink(missing_ok=True)
    return output_path
