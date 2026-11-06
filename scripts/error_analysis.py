from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from PIL import Image

from roadguard.constants import CLASS_CODES, CLASS_NAMES
from roadguard.inference.engine import RoadDamageDetector


def box_iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def load_ground_truth(label_path: Path, width: int, height: int) -> list[dict]:
    ground_truth = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id_text, x_center, y_center, box_width, box_height = line.split()
        class_id = int(class_id_text)
        x_center_px = float(x_center) * width
        y_center_px = float(y_center) * height
        width_px = float(box_width) * width
        height_px = float(box_height) * height
        ground_truth.append(
            {
                "class_id": class_id,
                "box": [
                    x_center_px - width_px / 2,
                    y_center_px - height_px / 2,
                    x_center_px + width_px / 2,
                    y_center_px + height_px / 2,
                ],
            }
        )
    return ground_truth


def match_detections(
    ground_truth: list[dict], predictions: list[dict]
) -> tuple[list[dict], int, int]:
    unmatched_predictions = set(range(len(predictions)))
    matches = []
    for truth in ground_truth:
        candidates = [
            (index, box_iou(truth["box"], predictions[index]["box"]))
            for index in unmatched_predictions
            if predictions[index]["class_id"] == truth["class_id"]
        ]
        if not candidates:
            continue
        prediction_index, iou = max(candidates, key=lambda item: item[1])
        if iou < 0.5:
            continue
        unmatched_predictions.remove(prediction_index)
        matches.append(
            {
                "class_id": truth["class_id"],
                "iou": iou,
                "confidence": predictions[prediction_index]["confidence"],
            }
        )
    false_negatives = len(ground_truth) - len(matches)
    false_positives = len(predictions) - len(matches)
    return matches, false_positives, false_negatives


def analyse(
    weights: Path,
    sample_root: Path,
    output_path: Path,
    examples_dir: Path,
    *,
    confidence: float,
) -> dict:
    detector = RoadDamageDetector(weights)
    image_paths = sorted(
        path
        for path in sample_root.rglob("*")
        if path.parent.name == "images" and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise ValueError(f"No representative images found under {sample_root}")
    examples_dir.mkdir(parents=True, exist_ok=True)
    class_truth = Counter()
    class_matches = Counter()
    matched_ious: list[float] = []
    total_predictions = 0
    total_truth = 0
    total_false_positives = 0
    total_false_negatives = 0
    records = []
    for image_path in image_paths:
        label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
        with Image.open(image_path) as opened:
            width, height = opened.size
        truth = load_ground_truth(label_path, width, height)
        prediction, annotated = detector.predict(image_path, confidence=confidence)
        predicted = [
            {
                "class_id": detection.class_id,
                "confidence": detection.confidence,
                "box": [detection.x1, detection.y1, detection.x2, detection.y2],
            }
            for detection in prediction.detections
        ]
        matches, false_positives, false_negatives = match_detections(truth, predicted)
        class_truth.update(item["class_id"] for item in truth)
        class_matches.update(item["class_id"] for item in matches)
        matched_ious.extend(item["iou"] for item in matches)
        total_truth += len(truth)
        total_predictions += len(predicted)
        total_false_positives += false_positives
        total_false_negatives += false_negatives
        annotated.save(examples_dir / f"{image_path.parent.parent.name}__{image_path.name}")
        records.append(
            {
                "image": str(image_path.relative_to(sample_root)),
                "ground_truth": len(truth),
                "predictions": len(predicted),
                "true_positives_at_iou_0_5": len(matches),
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "matched_mean_iou": (
                    statistics.fmean(item["iou"] for item in matches) if matches else None
                ),
            }
        )
    true_positives = len(matched_ious)
    payload = {
        "scope": "representative 24-image sample; not the frozen test-set evaluator",
        "weights": str(weights),
        "images_evaluated": len(image_paths),
        "confidence_threshold": confidence,
        "match_iou_threshold": 0.5,
        "ground_truth_instances": total_truth,
        "predicted_instances": total_predictions,
        "true_positives": true_positives,
        "false_positives": total_false_positives,
        "false_negatives": total_false_negatives,
        "sample_precision": (
            true_positives / (true_positives + total_false_positives)
            if true_positives + total_false_positives
            else 0.0
        ),
        "sample_recall": true_positives / total_truth if total_truth else 0.0,
        "mean_matched_iou": statistics.fmean(matched_ious) if matched_ious else None,
        "per_class": [
            {
                "class_id": class_id,
                "class_code": CLASS_CODES[class_id],
                "class_name": CLASS_NAMES[class_id],
                "ground_truth_instances": class_truth[class_id],
                "matched_instances": class_matches[class_id],
            }
            for class_id in range(4)
        ],
        "images": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    print(
        json.dumps(
            analyse(
                args.weights,
                args.samples,
                args.output,
                args.examples,
                confidence=args.confidence,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
