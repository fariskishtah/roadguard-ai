# RoadGuard AI

**Intelligent Road Damage Detection & Infrastructure Inspection Platform**

RoadGuard AI is an end-to-end object-detection project for the four verified RDD2022 road-damage categories: D00 longitudinal crack, D10 transverse crack, D20 alligator crack, and D40 pothole. It is owned by Faris Mohamed Kishtah and is being built as Sprint 2 Project 2 of the INSTANT AI & Data Science Training program.

> Current status: active implementation and genuine experiment execution. No benchmark numbers are published until the corresponding checkpoints and metric artifacts exist.

## Reproducibility

```bash
uv sync --extra dev
uv run pytest
uv run python scripts/audit_dataset.py data/raw/extracted
uv run python scripts/prepare_dataset.py data/raw/extracted data/processed/rdd2022
```

Training uses three distinct Ultralytics-supported detector families with comparable nano capacity: `yolov8n.pt`, `yolo11n.pt`, and `yolo26n.pt`. See `PROJECT_STATUS.md` for evidence paths and current blockers.

## Dataset and licensing

RDD2022 is sourced only from the [official RoadDamageDetector repository](https://github.com/sekilab/RoadDamageDetector) and its linked official object storage. The challenge documents 41,500 annotated training images and 6,000 unannotated challenge-test images across India, Japan, Czech Republic, Norway, the United States, and China. The repository states that dataset images are CC BY-SA 4.0. Raw data is excluded from Git and is never redistributed by this repository.

This project uses Ultralytics under AGPL-3.0 and is correspondingly released under **AGPL-3.0-only**. It is a research and portfolio application, not a safety-critical or autonomous-driving system.

## Application

After a trained production checkpoint is placed at `models/roadguard-production.pt`:

```bash
uv run python app.py
```

The local Gradio app provides real image inference, confidence and NMS controls, annotated output, structured detection tables, session analytics, JSON/CSV/image export, and a PDF inspection report. Missing weights produce an explicit error; the UI never substitutes fake predictions.

