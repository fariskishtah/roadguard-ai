# RoadGuard AI — Project Status

Last updated: 2026-11-06 (Africa/Cairo)

Status: **PARTIAL — implementation and real experiments in progress**

## Completed

- Independent Git repository created at `/Users/fariskishtah/project/roadguard-ai`.
- Official RDD2022 source, taxonomy, country counts, Pascal VOC format, and absence of public challenge-test annotations verified.
- Official country-specific downloads identified; raw data is Git-ignored.
- Ultralytics support verified for three distinct families: YOLOv8, YOLO11, YOLO26.
- AGPL-3.0 public-project licensing path selected; dataset attribution recorded as CC BY-SA 4.0.
- Pascal VOC parsing, bounding-box validation/clipping, YOLO normalization, deterministic country-stratified splits, namespacing, model training, test evaluation, benchmarking, inference, rendering, exports, PDF reporting, and Gradio UI implemented.
- Public GitHub repository created: `https://github.com/fariskishtah/roadguard-ai`.
- GitHub recognizes the repository as AGPL-3.0; unauthenticated HTTP verification returned 200.
- GitHub Actions CI run `42136962256` passed installation, Ruff, and all tests.
- Local test suite: 14 passed; no tests skipped.
- Secret/token/private-key scan found no credential material; raw data and weights are Git-ignored.

## Evidence and fixed configuration

- Seed: `42`
- Classes: `0 D00`, `1 D10`, `2 D20`, `3 D40`
- Primary image size: `640`
- Candidate checkpoints: `yolov8n.pt`, `yolo11n.pt`, `yolo26n.pt`
- Primary selection metric: validation mAP@0.50:0.95
- Frozen split manifest (after data preparation): `data/processed/rdd2022/split_manifest.json`
- Dataset audit (after download): `artifacts/metrics/dataset_audit.json`
- Training runs: `runs/train/`
- Genuine metrics: `artifacts/metrics/`
- Production checkpoint target: `models/roadguard-production.pt`

## Running processes

- Kaggle kernel: `fariskishtah/roadguard-ai-three-yolo-architectures`
- URL: `https://www.kaggle.com/code/fariskishtah/roadguard-ai-three-yolo-architectures`
- Version 1 status: `ERROR` — the attached dataset mount path differed from the dataset slug; no training ran.
- Version 2 status: `ERROR` — confirmed Kaggle's new nested mount root is `/kaggle/input/datasets/...`; no training ran.
- Version 3: recursive input discovery added before any training.
- Version 4: metric parameter counting corrected to use the loaded YOLO model object.
- Version 5: per-class metrics corrected to use Ultralytics `class_result()` API.
- Versions 3 and 4 were explicitly cancelled by their exact Kaggle session IDs after confirming they were stuck traversing the nested input mount and would later hit the known metric API error. No useful training artifacts existed.
- Version 5 uses bounded input glob patterns and makes ONNX export non-fatal to preserve successful training/evaluation evidence.
- Version 5 status: explicitly cancelled after confirming full-mirror small-file label I/O was consuming GPU without reaching smoke training.
- Version 6 audits every label, box, resolution, and hash in the actual 6,000-image subset while retaining inventory counts for the larger transport folders.

## Not yet complete

- Official subset download and measured audit.
- Kaggle smoke run, three full training runs, controlled tuning, and held-out evaluation.
- Real checkpoint integration tests and export parity.
- Genuine error-analysis figures and application screenshots.
- Security/license release audit, public GitHub/model/Space publication, and unauthenticated live verification.

No metrics or completion claims are placeholders.

