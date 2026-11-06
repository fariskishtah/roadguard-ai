"""Reproducible RoadGuard AI Kaggle GPU experiment.

The attached Kaggle mirror is used only as transport for RDD2022 files. This
script validates labels, rebuilds a group-aware frozen split, tunes on validation
only, selects the production model before test evaluation, and exports evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "ultralytics==8.4.203",
        "pyyaml>=6,<7",
    ],
    check=True,
)

import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics import __version__ as ultralytics_version  # noqa: E402

SEED = 42
MAX_PER_COUNTRY = 1000
IMG_SIZE = 640
BATCH = 16
FINAL_EPOCHS = 20
CLASS_CODES = ["D00", "D10", "D20", "D40"]
CLASS_NAMES = ["Longitudinal Crack", "Transverse Crack", "Alligator Crack", "Pothole"]
MODELS = ["yolov8n.pt", "yolo11n.pt", "yolo26n.pt"]
INPUT_BASE = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working/roadguard")
DATA_ROOT = WORK_ROOT / "dataset"
RUNS_ROOT = WORK_ROOT / "runs"
ARTIFACTS_ROOT = WORK_ROOT / "artifacts"
OUTPUT_ROOT = Path("/kaggle/working/roadguard-output")

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def locate_input_root() -> Path:
    candidates = [
        path.parent
        for path in INPUT_BASE.rglob("data.yaml")
        if (path.parent / "train" / "images").is_dir()
    ]
    if not candidates:
        inventory = [str(path) for path in INPUT_BASE.iterdir()] if INPUT_BASE.exists() else []
        raise FileNotFoundError(
            f"RDD2022 Kaggle input was not mounted; input inventory: {inventory}"
        )
    return candidates[0]


def country_from_stem(stem: str) -> str:
    lowered = stem.lower().replace("-", "_")
    prefixes = {
        "Japan": ("japan",),
        "India": ("india",),
        "Czech": ("czech",),
        "Norway": ("norway",),
        "United_States": ("united_states", "unitedstates", "usa"),
        "China": ("china",),
    }
    for country, values in prefixes.items():
        if lowered.startswith(values):
            return country
    return "Unknown"


def sequence_group(country: str, stem: str) -> str:
    numbers = re.findall(r"\d+", stem)
    if not numbers:
        return f"{country}:{stem}"
    return f"{country}:{int(numbers[-1]) // 10:06d}"


def parse_label(path: Path) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    boxes = []
    errors = []
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        parts = raw.split()
        if len(parts) != 5:
            errors.append(f"{path}:{line_number}:field_count")
            continue
        try:
            class_id = int(parts[0])
            coords = tuple(float(value) for value in parts[1:])
        except ValueError:
            errors.append(f"{path}:{line_number}:parse")
            continue
        if class_id not in range(4):
            errors.append(f"{path}:{line_number}:class_{class_id}")
            continue
        if not all(0.0 <= value <= 1.0 for value in coords) or coords[2] <= 0 or coords[3] <= 0:
            errors.append(f"{path}:{line_number}:coordinates")
            continue
        boxes.append((class_id, *coords))
    return boxes, errors


def discover_records(input_root: Path) -> tuple[list[dict], dict]:
    records = []
    full_counts = Counter()
    full_classes = Counter()
    empty = 0
    invalid = []
    # The mirror's test folder mixes a labeled holdout with the official 6,000-image
    # unannotated challenge test set. Restrict discovery to its train+val transport
    # folders, then construct a new independent train/val/test split below.
    for source_split in ("train", "val"):
        image_dir = input_root / source_split / "images"
        label_dir = input_root / source_split / "labels"
        if not image_dir.exists():
            continue
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            # RDD2022 challenge-test images have no public annotations.
            if not label_path.exists():
                continue
            boxes, errors = parse_label(label_path)
            invalid.extend(errors)
            country = country_from_stem(image_path.stem)
            full_counts[country] += 1
            full_classes.update(box[0] for box in boxes)
            empty += int(not boxes)
            records.append(
                {
                    "country": country,
                    "stem": image_path.stem,
                    "image": image_path,
                    "label": label_path,
                    "boxes": boxes,
                    "group": sequence_group(country, image_path.stem),
                    "source_split": source_split,
                }
            )
    audit = {
        "transport_mirror": "sreekaraditya/rdd2022-yolo-crackscan-v2",
        "mounted_input_root": str(input_root),
        "labeled_images_discovered": len(records),
        "images_by_country": dict(full_counts),
        "annotations_by_class_id": {str(k): v for k, v in full_classes.items()},
        "empty_label_images": empty,
        "invalid_label_entries": invalid,
        "transport_test_excluded": True,
        "transport_test_exclusion_reason": (
            "It mixes public labeled data with the official unannotated challenge test set."
        ),
    }
    return records, audit


def select_subset(records: list[dict]) -> list[dict]:
    by_country = defaultdict(list)
    for record in records:
        if record["country"] != "Unknown":
            by_country[record["country"]].append(record)
    selected = []
    rng = random.Random(SEED)
    for country in sorted(by_country):
        candidates = sorted(by_country[country], key=lambda item: item["stem"])
        rng.shuffle(candidates)
        selected.extend(candidates[:MAX_PER_COUNTRY])
    return sorted(selected, key=lambda item: (item["country"], item["stem"]))


def split_for_group(group: str) -> str:
    digest = hashlib.sha256(f"{SEED}|{group}".encode()).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    if value < 0.15:
        return "test"
    if value < 0.30:
        return "val"
    return "train"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_dataset(selected: list[dict], audit: dict) -> Path:
    manifest = {"seed": SEED, "grouping": "country + numeric frame bucket of 10", "records": []}
    split_counts = Counter()
    country_split_counts = Counter()
    class_counts = Counter()
    image_hashes = defaultdict(list)
    hash_source_groups = defaultdict(list)
    resolution_counts = Counter()
    for record in selected:
        record["sha256"] = sha256_file(record["image"])
        image_hashes[record["sha256"]].append(f"{record['country']}__{record['stem']}")
        hash_source_groups[record["sha256"]].append(record["group"])
    # Exact duplicates inherit one common grouping key before split assignment.
    hash_group_leaders = {digest: min(groups) for digest, groups in hash_source_groups.items()}
    for record in selected:
        split_group = hash_group_leaders[record["sha256"]]
        split = split_for_group(split_group)
        image_dir = DATA_ROOT / "images" / split
        label_dir = DATA_ROOT / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        unique_stem = f"{record['country']}__{record['stem']}"
        target_image = image_dir / f"{unique_stem}{record['image'].suffix.lower()}"
        target_label = label_dir / f"{unique_stem}.txt"
        os.symlink(record["image"], target_image)
        shutil.copy2(record["label"], target_label)
        with Image.open(record["image"]) as image:
            resolution_counts[f"{image.width}x{image.height}"] += 1
        split_counts[split] += 1
        country_split_counts[f"{record['country']}|{split}"] += 1
        class_counts.update(box[0] for box in record["boxes"])
        manifest["records"].append(
            {
                "sample_id": unique_stem,
                "country": record["country"],
                "group": record["group"],
                "split_group": split_group,
                "split": split,
                "source_split": record["source_split"],
                "sha256": record["sha256"],
            }
        )
    duplicates = [values for values in image_hashes.values() if len(values) > 1]
    record_by_id = {record["sample_id"]: record for record in manifest["records"]}
    leakage = [
        group
        for group in duplicates
        if len({record_by_id[sample]["split"] for sample in group}) > 1
    ]
    audit.update(
        {
            "selected_images": len(selected),
            "selected_annotations_by_class_id": {str(k): v for k, v in class_counts.items()},
            "split_counts": dict(split_counts),
            "country_split_counts": dict(country_split_counts),
            "resolution_counts": dict(resolution_counts),
            "exact_duplicate_groups": duplicates,
            "exact_duplicate_split_leakage": leakage,
            "geographic_independence_claimed": False,
            "leakage_note": (
                "Adjacent numeric frames are grouped and exact hashes audited. Route/location IDs "
                "are unavailable, so complete geographic independence is not claimed."
            ),
        }
    )
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_ROOT / "dataset_audit.json").write_text(json.dumps(audit, indent=2))
    (ARTIFACTS_ROOT / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    data_yaml = {
        "path": str(DATA_ROOT),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(CLASS_NAMES)),
    }
    yaml_path = DATA_ROOT / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    return yaml_path


def train_once(
    model_name: str,
    data_yaml: Path,
    run_name: str,
    *,
    epochs: int,
    image_size: int,
    batch: int,
    learning_rate: float,
    fraction: float = 1.0,
    plots: bool = True,
) -> tuple[Path, float]:
    started = time.perf_counter()
    result = YOLO(model_name).train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        seed=SEED,
        deterministic=True,
        optimizer="auto",
        lr0=learning_rate,
        weight_decay=0.0005,
        patience=max(8, epochs // 2),
        cos_lr=True,
        close_mosaic=min(5, max(1, epochs // 4)),
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.35,
        degrees=2.0,
        translate=0.08,
        scale=0.35,
        shear=1.0,
        perspective=0.0002,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.8,
        mixup=0.0,
        fraction=fraction,
        project=str(RUNS_ROOT),
        name=run_name,
        exist_ok=True,
        device=0,
        workers=4,
        plots=plots,
        save=True,
        verbose=True,
    )
    return Path(result.save_dir) / "weights" / "best.pt", time.perf_counter() - started


def compact_metrics(metrics, weights: Path, split: str, parameters: int) -> dict:
    box = metrics.box
    return {
        "weights": str(weights),
        "split": split,
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "per_class": [
            {
                "class_id": index,
                "class_code": CLASS_CODES[index],
                "class_name": CLASS_NAMES[index],
                "precision": float(box.class_result(index)[0]),
                "recall": float(box.class_result(index)[1]),
                "ap50": float(box.class_result(index)[2]),
                "ap50_95": float(box.class_result(index)[3]),
            }
            for index in range(4)
        ],
        "speed_ms": {key: float(value) for key, value in metrics.speed.items()},
        "parameters": parameters,
        "size_mb": weights.stat().st_size / (1024 * 1024),
    }


def validate(weights: Path, data_yaml: Path, split: str, run_name: str) -> dict:
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=0,
        workers=4,
        plots=True,
        project=str(RUNS_ROOT / "evaluation"),
        name=run_name,
        exist_ok=True,
        verbose=False,
    )
    parameters = sum(parameter.numel() for parameter in model.model.parameters())
    return compact_metrics(metrics, weights, split, parameters)


def benchmark(weights: Path, image_paths: list[Path]) -> dict:
    model = YOLO(str(weights))
    prepared = []
    for path in image_paths:
        with Image.open(path) as image:
            prepared.append(image.convert("RGB").copy())
    for image in prepared[:5]:
        model.predict(image, imgsz=IMG_SIZE, device=0, verbose=False)
    torch.cuda.synchronize()
    model_only = []
    end_to_end = []
    for index in range(50):
        image = prepared[index % len(prepared)]
        total_start = time.perf_counter()
        torch.cuda.synchronize()
        model_start = time.perf_counter()
        model.predict(image, imgsz=IMG_SIZE, device=0, verbose=False)
        torch.cuda.synchronize()
        model_only.append((time.perf_counter() - model_start) * 1000)
        end_to_end.append((time.perf_counter() - total_start) * 1000)

    def summarize(samples: list[float]) -> dict:
        ordered = sorted(samples)
        median = statistics.median(ordered)
        return {
            "median_ms": median,
            "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
            "fps": 1000.0 / median,
        }

    return {
        "device": torch.cuda.get_device_name(0),
        "image_size": IMG_SIZE,
        "batch_size": 1,
        "warmup": 5,
        "repetitions": 50,
        "model_only": summarize(model_only),
        "end_to_end": summarize(end_to_end),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU is not enabled; refusing full experiments on CPU")
    for path in (WORK_ROOT, DATA_ROOT, RUNS_ROOT, ARTIFACTS_ROOT, OUTPUT_ROOT):
        path.mkdir(parents=True, exist_ok=True)
    input_root = locate_input_root()
    records, audit = discover_records(input_root)
    if len(records) < 30000:
        raise RuntimeError(f"Expected at least 30,000 labeled images, found {len(records)}")
    if audit["invalid_label_entries"]:
        raise RuntimeError(f"Invalid labels found: {audit['invalid_label_entries'][:10]}")
    selected = select_subset(records)
    data_yaml = prepare_dataset(selected, audit)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": ultralytics_version,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "seed": SEED,
        "image_size": IMG_SIZE,
        "batch": BATCH,
        "final_epochs": FINAL_EPOCHS,
    }
    (ARTIFACTS_ROOT / "environment.json").write_text(json.dumps(environment, indent=2))

    smoke_results = []
    for model_name in MODELS:
        checkpoint, duration = train_once(
            model_name,
            data_yaml,
            f"smoke_{Path(model_name).stem}",
            epochs=1,
            image_size=320,
            batch=16,
            learning_rate=0.01,
            fraction=0.05,
            plots=False,
        )
        smoke_results.append(
            {"model": model_name, "checkpoint": str(checkpoint), "duration_seconds": duration}
        )
    (ARTIFACTS_ROOT / "smoke_results.json").write_text(json.dumps(smoke_results, indent=2))

    tuning_results = []
    for learning_rate in (0.003, 0.01, 0.02):
        slug = str(learning_rate).replace(".", "_")
        checkpoint, duration = train_once(
            "yolo11n.pt",
            data_yaml,
            f"tune_yolo11n_lr_{slug}",
            epochs=5,
            image_size=512,
            batch=BATCH,
            learning_rate=learning_rate,
            fraction=0.35,
            plots=False,
        )
        result = validate(checkpoint, data_yaml, "val", f"tune_yolo11n_lr_{slug}_val")
        tuning_results.append(
            {
                "experiment_id": f"yolo11n-lr-{learning_rate}",
                "learning_rate": learning_rate,
                "epochs": 5,
                "fraction": 0.35,
                "duration_seconds": duration,
                "checkpoint": str(checkpoint),
                "validation": result,
            }
        )
    best_tuning = max(tuning_results, key=lambda item: item["validation"]["map50_95"])
    selected_yolo11_lr = best_tuning["learning_rate"]
    (ARTIFACTS_ROOT / "tuning_results.json").write_text(json.dumps(tuning_results, indent=2))

    final_runs = []
    validation_results = []
    checkpoints = {}
    for model_name in MODELS:
        learning_rate = selected_yolo11_lr if model_name == "yolo11n.pt" else 0.01
        run_name = f"final_{Path(model_name).stem}"
        checkpoint, duration = train_once(
            model_name,
            data_yaml,
            run_name,
            epochs=FINAL_EPOCHS,
            image_size=IMG_SIZE,
            batch=BATCH,
            learning_rate=learning_rate,
        )
        checkpoints[model_name] = checkpoint
        final_runs.append(
            {
                "model": model_name,
                "checkpoint": str(checkpoint),
                "learning_rate": learning_rate,
                "duration_seconds": duration,
                "hardware": torch.cuda.get_device_name(0),
            }
        )
        result = validate(checkpoint, data_yaml, "val", f"{run_name}_val")
        result["model"] = model_name
        validation_results.append(result)
    (ARTIFACTS_ROOT / "final_runs.json").write_text(json.dumps(final_runs, indent=2))
    (ARTIFACTS_ROOT / "validation_results.json").write_text(
        json.dumps(validation_results, indent=2)
    )

    production_validation = max(validation_results, key=lambda item: item["map50_95"])
    production_model = production_validation["model"]
    selection = {
        "selected_model": production_model,
        "criterion": "highest validation mAP@0.50:0.95",
        "selected_validation_metrics": production_validation,
        "decision_frozen_before_test_evaluation": True,
    }
    (ARTIFACTS_ROOT / "production_selection.json").write_text(json.dumps(selection, indent=2))

    test_results = []
    comparison = []
    test_images = sorted((DATA_ROOT / "images" / "test").iterdir())[:30]
    for model_name in MODELS:
        checkpoint = checkpoints[model_name]
        result = validate(checkpoint, data_yaml, "test", f"final_{Path(model_name).stem}_test")
        latency = benchmark(checkpoint, test_images)
        result["model"] = model_name
        result["benchmark"] = latency
        test_results.append(result)
        comparison.append(
            {
                "model": Path(model_name).stem,
                "map50": result["map50"],
                "map50_95": result["map50_95"],
                "precision": result["precision"],
                "recall": result["recall"],
                "fps": latency["model_only"]["fps"],
                "median_latency_ms": latency["model_only"]["median_ms"],
                "p95_latency_ms": latency["model_only"]["p95_ms"],
                "size_mb": result["size_mb"],
                "parameters": result["parameters"],
            }
        )
    (ARTIFACTS_ROOT / "test_results.json").write_text(json.dumps(test_results, indent=2))
    (ARTIFACTS_ROOT / "model_comparison.json").write_text(json.dumps(comparison, indent=2))

    for model_name, checkpoint in checkpoints.items():
        shutil.copy2(checkpoint, OUTPUT_ROOT / f"{Path(model_name).stem}-rdd2022.pt")
    production_checkpoint = checkpoints[production_model]
    production_target = OUTPUT_ROOT / "roadguard-production.pt"
    shutil.copy2(production_checkpoint, production_target)
    selection["production_checkpoint_sha256"] = sha256_file(production_target)
    selection["production_checkpoint"] = str(production_target)
    (ARTIFACTS_ROOT / "production_selection.json").write_text(json.dumps(selection, indent=2))
    shutil.copytree(ARTIFACTS_ROOT, OUTPUT_ROOT / "metrics", dirs_exist_ok=True)

    onnx_path = Path(
        YOLO(str(production_target)).export(
            format="onnx", imgsz=IMG_SIZE, simplify=True, dynamic=False, device=0
        )
    )
    shutil.copy2(onnx_path, OUTPUT_ROOT / "roadguard-production.onnx")
    print(
        json.dumps(
            {"status": "complete", "selection": selection, "comparison": comparison}, indent=2
        )
    )


if __name__ == "__main__":
    main()
