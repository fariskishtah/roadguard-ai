# RoadGuard AI model card

Status: awaiting completed Kaggle experiment artifacts. This file intentionally contains no placeholder metrics.

## Intended use

Research and portfolio demonstration of bounding-box detection for four RDD2022 road-damage categories. The detector may support human infrastructure inspection workflows after domain-specific review.

## Out-of-scope use

- Autonomous-driving decisions.
- Safety-critical alerts without human review.
- Physical severity, repair-cost, depth, area, or GPS estimation.
- Claims of global generalization without country-specific evaluation.

## Training and evaluation protocol

Three distinct nano-scale families—YOLOv8, YOLO11, and YOLO26—receive the same frozen subset and image resolution. Hyperparameters are tuned on validation only. The production checkpoint is selected by validation mAP@0.50:0.95 before all three models are evaluated on the untouched custom test split. Latency uses batch size 1, fixed 640-pixel input, five warmups, and fifty repeated measurements on one documented device.

## Limitations

Expected failure modes include small cracks, shadows, lane markings, wet pavement, poor light, unusual texture, oblique/top-down domain shift, and confusion between longitudinal and transverse cracks. A trained checkpoint does not certify road condition or replace a qualified inspector.

