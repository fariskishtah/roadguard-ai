from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from functools import lru_cache
from pathlib import Path

import gradio as gr
import pandas as pd
from PIL import Image

from roadguard.inference.engine import RoadDamageDetector
from roadguard.inference.export import export_csv, export_json, prediction_rows
from roadguard.inference.processing import process_batch, process_video
from roadguard.inference.report import create_pdf_report

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "roadguard-production.pt"


@lru_cache(maxsize=1)
def get_detector() -> RoadDamageDetector:
    weights = Path(os.environ.get("ROADGUARD_WEIGHTS", DEFAULT_WEIGHTS))
    return RoadDamageDetector(weights)


def _summary(prediction) -> str:
    counts = Counter(det.damage_type for det in prediction.detections)
    average = (
        sum(det.confidence for det in prediction.detections) / len(prediction.detections)
        if prediction.detections
        else 0.0
    )
    breakdown = (
        " · ".join(f"{key}: {value}" for key, value in counts.items()) or "No damage detected"
    )
    return (
        f"### {len(prediction.detections)} detections\n"
        f"{breakdown}  \n"
        f"Average confidence: **{average:.1%}** · "
        f"Inference: **{prediction.inference_ms:.1f} ms** · "
        f"Model: **{prediction.model_version}**"
    )


def detect_image(image: Image.Image | None, confidence: float, nms_iou: float):
    if image is None:
        raise gr.Error("Upload a supported road image first.")
    try:
        prediction, annotated = get_detector().predict(
            image, confidence=confidence, nms_iou=nms_iou, filename="uploaded-image"
        )
    except (FileNotFoundError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc
    rows = prediction_rows(prediction)
    frame = pd.DataFrame(rows)
    work = Path(tempfile.mkdtemp(prefix="roadguard-report-"))
    json_path = export_json(prediction, work / "detections.json")
    csv_path = export_csv(prediction, work / "detections.csv")
    image_path = work / "annotated.jpg"
    annotated.save(image_path, quality=92)
    report_path = create_pdf_report(prediction, annotated, work / "inspection-report.pdf")
    return (
        annotated,
        _summary(prediction),
        frame,
        str(json_path),
        str(csv_path),
        str(image_path),
        str(report_path),
        prediction.model_dump(),
    )


def update_session(prediction: dict | None, history: list | None):
    history = list(history or [])
    if prediction:
        history.append(prediction)
    detections = sum(len(item["detections"]) for item in history)
    times = [item["inference_ms"] for item in history]
    classes = Counter(det["damage_type"] for item in history for det in item.get("detections", []))
    return history, {
        "processed_images": len(history),
        "total_detections": detections,
        "average_inference_ms": sum(times) / len(times) if times else 0,
        "class_distribution": dict(classes),
    }


def run_batch(files, confidence: float, nms_iou: float):
    paths = [item if isinstance(item, str) else item.name for item in (files or [])]
    try:
        archive, summary = process_batch(
            get_detector(), paths, confidence=confidence, nms_iou=nms_iou
        )
    except (FileNotFoundError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc
    return str(archive), summary


def run_video(video, confidence: float, nms_iou: float):
    if not video:
        raise gr.Error("Upload a supported short road video first.")
    path = video if isinstance(video, str) else video.name
    try:
        output, summary = process_video(
            get_detector(), path, confidence=confidence, nms_iou=nms_iou
        )
    except (FileNotFoundError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc
    return str(output), summary


CSS = """
:root { --rg-navy:#071426; --rg-blue:#1687ff; --rg-orange:#ff9f43; }
.gradio-container { max-width: 1440px !important; background: #f5f8fc; }
.hero { padding: 28px 32px; border-radius: 20px; color: white;
  background: radial-gradient(circle at 85% 20%, #174a7e 0, #0b2748 35%, #071426 75%); }
.hero h1 { margin: 0; font-size: 2.5rem; letter-spacing: -.04em; }
.hero p { color: #bdd0e5; font-size: 1.05rem; max-width: 780px; }
.metric-card { border: 1px solid #dce6f2; border-radius: 14px; }
footer { display:none !important; }
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(css=CSS, title="RoadGuard AI", theme=gr.themes.Soft()) as demo:
        session_history = gr.State([])
        current_prediction = gr.State(None)
        gr.HTML(
            "<section class='hero'><h1>RoadGuard AI</h1>"
            "<p>AI-powered road damage detection and infrastructure inspection. "
            "Every box, confidence score, and summary is produced by the deployed "
            "detector.</p></section>"
        )
        with gr.Tabs():
            with gr.Tab("Image inspection"):
                with gr.Row():
                    with gr.Column(scale=1):
                        source = gr.Image(type="pil", label="Road image")
                        confidence = gr.Slider(
                            0.05, 0.95, value=0.25, step=0.05, label="Confidence threshold"
                        )
                        nms_iou = gr.Slider(
                            0.1, 0.9, value=0.45, step=0.05, label="NMS IoU threshold"
                        )
                        run = gr.Button("Run road inspection", variant="primary")
                    with gr.Column(scale=1):
                        annotated = gr.Image(type="pil", label="Annotated result")
                        summary = gr.Markdown("Upload an image to begin.")
                table = gr.Dataframe(label="Detections", interactive=False)
                with gr.Row():
                    json_file = gr.File(label="JSON")
                    csv_file = gr.File(label="CSV")
                    image_file = gr.File(label="Annotated image")
                    pdf_file = gr.File(label="PDF report")
                outputs = [
                    annotated,
                    summary,
                    table,
                    json_file,
                    csv_file,
                    image_file,
                    pdf_file,
                    current_prediction,
                ]
                analytics = gr.JSON(label="Session analytics")
                run.click(detect_image, [source, confidence, nms_iou], outputs).then(
                    update_session,
                    [current_prediction, session_history],
                    [session_history, analytics],
                )
            with gr.Tab("Batch images"):
                batch_files = gr.File(
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                    label="Up to 20 road images",
                )
                with gr.Row():
                    batch_confidence = gr.Slider(0.05, 0.95, value=0.25, step=0.05)
                    batch_iou = gr.Slider(0.1, 0.9, value=0.45, step=0.05)
                batch_run = gr.Button("Process batch", variant="primary")
                batch_archive = gr.File(label="Download batch results")
                batch_summary = gr.Dataframe(label="Per-image summary", interactive=False)
                batch_run.click(
                    run_batch,
                    [batch_files, batch_confidence, batch_iou],
                    [batch_archive, batch_summary],
                )
            with gr.Tab("Video inspection"):
                gr.Markdown("Short videos up to 90 seconds; output is capped at 1280 px wide.")
                video_source = gr.Video(label="Road video")
                with gr.Row():
                    video_confidence = gr.Slider(0.05, 0.95, value=0.25, step=0.05)
                    video_iou = gr.Slider(0.1, 0.9, value=0.45, step=0.05)
                video_run = gr.Button("Process video", variant="primary")
                video_output = gr.Video(label="Annotated video")
                video_summary = gr.JSON(label="Processing summary")
                video_run.click(
                    run_video,
                    [video_source, video_confidence, video_iou],
                    [video_output, video_summary],
                )
            with gr.Tab("Model comparison"):
                metrics_path = PROJECT_ROOT / "artifacts" / "metrics" / "model_comparison.json"
                if metrics_path.exists():
                    comparison = pd.DataFrame(json.loads(metrics_path.read_text()))
                    gr.Dataframe(comparison, interactive=False)
                else:
                    gr.Markdown(
                        "Comparison metrics will appear here only after all three "
                        "checkpoints are evaluated "
                        "on the frozen held-out test split. No placeholder results are shown."
                    )
            with gr.Tab("About & limitations"):
                gr.Markdown(
                    """### Four verified RDD2022 categories

- **D00** — Longitudinal Crack
- **D10** — Transverse Crack
- **D20** — Alligator Crack
- **D40** — Pothole

RoadGuard AI is an inspection-support research project, not an autonomous-driving or
safety-critical system. Predictions can be wrong, especially under domain shift, poor lighting,
wet pavement, shadows, unusual camera angles, and for very small damage regions.
"""
                )
    return demo


demo = build_app()

if __name__ == "__main__":
    demo.launch()
