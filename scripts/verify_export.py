from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from ultralytics import YOLO


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


def detections(result) -> list[dict]:
    if result.boxes is None:
        return []
    return [
        {"box": box, "class_id": int(class_id), "confidence": float(confidence)}
        for box, class_id, confidence in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            strict=True,
        )
    ]


def verify_export(
    pytorch_weights: Path,
    exported_weights: Path,
    image_paths: list[Path],
    output_path: Path,
    *,
    confidence: float = 0.25,
    nms_iou: float = 0.45,
    image_size: int = 640,
) -> dict:
    if not image_paths:
        raise ValueError("At least one representative image is required")
    pytorch_model = YOLO(str(pytorch_weights))
    exported_model = YOLO(str(exported_weights), task="detect")
    matched_ious: list[float] = []
    confidence_differences: list[float] = []
    pytorch_latencies: list[float] = []
    exported_latencies: list[float] = []
    images = []
    total_pytorch = 0
    total_exported = 0
    total_matches = 0
    for path in image_paths:
        started = time.perf_counter()
        pytorch_result = pytorch_model.predict(
            path, conf=confidence, iou=nms_iou, imgsz=image_size, verbose=False
        )[0]
        pytorch_latencies.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        exported_result = exported_model.predict(
            path, conf=confidence, iou=nms_iou, imgsz=image_size, verbose=False
        )[0]
        exported_latencies.append((time.perf_counter() - started) * 1000)
        reference = detections(pytorch_result)
        candidate = detections(exported_result)
        total_pytorch += len(reference)
        total_exported += len(candidate)
        unmatched = set(range(len(candidate)))
        image_matches = []
        for reference_detection in reference:
            possible = [
                (index, box_iou(reference_detection["box"], candidate[index]["box"]))
                for index in unmatched
                if candidate[index]["class_id"] == reference_detection["class_id"]
            ]
            if not possible:
                continue
            best_index, best_iou = max(possible, key=lambda item: item[1])
            if best_iou < 0.5:
                continue
            unmatched.remove(best_index)
            total_matches += 1
            matched_ious.append(best_iou)
            confidence_difference = abs(
                reference_detection["confidence"] - candidate[best_index]["confidence"]
            )
            confidence_differences.append(confidence_difference)
            image_matches.append(
                {
                    "class_id": reference_detection["class_id"],
                    "iou": best_iou,
                    "confidence_difference": confidence_difference,
                }
            )
        images.append(
            {
                "image": path.name,
                "pytorch_detections": len(reference),
                "exported_detections": len(candidate),
                "matches_at_iou_0_5": len(image_matches),
                "matches": image_matches,
            }
        )
    payload = {
        "pytorch_weights": str(pytorch_weights),
        "exported_weights": str(exported_weights),
        "images_evaluated": len(image_paths),
        "confidence_threshold": confidence,
        "nms_iou_threshold": nms_iou,
        "image_size": image_size,
        "pytorch_detections": total_pytorch,
        "exported_detections": total_exported,
        "matched_detections_at_iou_0_5": total_matches,
        "reference_match_rate": total_matches / total_pytorch if total_pytorch else 1.0,
        "mean_matched_box_iou": statistics.fmean(matched_ious) if matched_ious else None,
        "mean_absolute_confidence_difference": (
            statistics.fmean(confidence_differences) if confidence_differences else None
        ),
        "median_pytorch_latency_ms": statistics.median(pytorch_latencies),
        "median_exported_latency_ms": statistics.median(exported_latencies),
        "images": images,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch", type=Path, required=True)
    parser.add_argument("--exported", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(
        path
        for path in args.images.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    print(
        json.dumps(
            verify_export(args.pytorch, args.exported, paths, args.output),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
