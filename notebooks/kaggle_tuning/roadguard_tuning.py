"""Controlled RoadGuard learning-rate tuning with an explicit AdamW optimizer."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "ultralytics==8.4.203"],
    check=True,
)

import torch  # noqa: E402
import yaml  # noqa: E402
from ultralytics import YOLO  # noqa: E402

SEED = 42
MAX_PER_COUNTRY = 1000
IMG_SIZE = 640
BATCH = 16
CLASS_CODES = ["D00", "D10", "D20", "D40"]
CLASS_NAMES = ["Longitudinal Crack", "Transverse Crack", "Alligator Crack", "Pothole"]
INPUT_BASE = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working/roadguard-tuning")
DATA_ROOT = WORK_ROOT / "dataset"
RUNS_ROOT = WORK_ROOT / "runs"
OUTPUT_ROOT = Path("/kaggle/working/roadguard-tuning-output")

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def locate_input() -> Path:
    patterns = ("*/data.yaml", "datasets/*/*/data.yaml", "datasets/*/*/versions/*/data.yaml")
    for pattern in patterns:
        for path in INPUT_BASE.glob(pattern):
            if (path.parent / "train" / "images").is_dir():
                return path.parent
    raise FileNotFoundError("RDD2022 mirror was not mounted")


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
    return f"{country}:{int(numbers[-1]) // 10:06d}" if numbers else f"{country}:{stem}"


def split_for_group(group: str) -> str:
    value = int(hashlib.sha256(f"{SEED}|{group}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if value < 0.15:
        return "test"
    if value < 0.30:
        return "val"
    return "train"


def prepare_dataset(input_root: Path) -> Path:
    by_country = defaultdict(list)
    for source_split in ("train", "val"):
        for image in sorted((input_root / source_split / "images").iterdir()):
            if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            country = country_from_stem(image.stem)
            if country == "Unknown":
                continue
            by_country[country].append(
                {
                    "country": country,
                    "stem": image.stem,
                    "image": image,
                    "label": input_root / source_split / "labels" / f"{image.stem}.txt",
                }
            )
    rng = random.Random(SEED)
    selected = []
    for country in sorted(by_country):
        candidates = sorted(by_country[country], key=lambda item: item["stem"])
        rng.shuffle(candidates)
        selected.extend(candidates[:MAX_PER_COUNTRY])
    split_counts = defaultdict(int)
    for record in selected:
        split = split_for_group(sequence_group(record["country"], record["stem"]))
        image_dir = DATA_ROOT / "images" / split
        label_dir = DATA_ROOT / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        unique_stem = f"{record['country']}__{record['stem']}"
        os.symlink(record["image"], image_dir / f"{unique_stem}{record['image'].suffix.lower()}")
        shutil.copy2(record["label"], label_dir / f"{unique_stem}.txt")
        split_counts[split] += 1
    if len(selected) != 6000:
        raise RuntimeError(f"Expected the frozen 6,000-image subset, found {len(selected)}")
    dataset = {
        "path": str(DATA_ROOT),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(CLASS_NAMES)),
    }
    path = DATA_ROOT / "dataset.yaml"
    path.write_text(yaml.safe_dump(dataset, sort_keys=False))
    (OUTPUT_ROOT / "split_counts.json").write_text(json.dumps(dict(split_counts), indent=2))
    return path


def train(
    data_yaml: Path,
    run_name: str,
    learning_rate: float,
    *,
    epochs: int,
    fraction: float,
    image_size: int,
) -> tuple[Path, float]:
    started = time.perf_counter()
    result = YOLO("yolo11n.pt").train(
        data=str(data_yaml),
        epochs=epochs,
        fraction=fraction,
        imgsz=image_size,
        batch=BATCH,
        seed=SEED,
        deterministic=True,
        optimizer="AdamW",
        lr0=learning_rate,
        lrf=0.01,
        weight_decay=0.0005,
        cos_lr=True,
        patience=max(8, epochs // 2),
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
        project=str(RUNS_ROOT),
        name=run_name,
        exist_ok=True,
        device=0,
        workers=4,
        plots=True,
        save=True,
        verbose=True,
    )
    return Path(result.save_dir) / "weights" / "best.pt", time.perf_counter() - started


def evaluate(weights: Path, data_yaml: Path, split: str, name: str) -> dict:
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
        name=name,
        exist_ok=True,
        verbose=False,
    )
    box = metrics.box
    return {
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
        "parameters": sum(parameter.numel() for parameter in model.model.parameters()),
        "size_mb": weights.stat().st_size / (1024 * 1024),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    data_yaml = prepare_dataset(locate_input())
    trials = []
    for learning_rate in (0.0005, 0.001, 0.003):
        slug = str(learning_rate).replace(".", "_")
        checkpoint, duration = train(
            data_yaml,
            f"adamw_lr_{slug}",
            learning_rate,
            epochs=6,
            fraction=0.35,
            image_size=512,
        )
        validation = evaluate(checkpoint, data_yaml, "val", f"adamw_lr_{slug}_val")
        trials.append(
            {
                "experiment_id": f"yolo11n-adamw-lr-{learning_rate}",
                "optimizer": "AdamW",
                "learning_rate": learning_rate,
                "epochs": 6,
                "fraction": 0.35,
                "image_size": 512,
                "duration_seconds": duration,
                "validation": validation,
            }
        )
    best = max(trials, key=lambda item: item["validation"]["map50_95"])
    final_checkpoint, duration = train(
        data_yaml,
        "final_tuned_yolo11n",
        best["learning_rate"],
        epochs=20,
        fraction=1.0,
        image_size=IMG_SIZE,
    )
    final_validation = evaluate(final_checkpoint, data_yaml, "val", "final_tuned_yolo11n_val")
    selection_record = {
        "selection_frozen_before_test": True,
        "selected_learning_rate": best["learning_rate"],
        "selected_by": "highest short-run validation mAP@0.50:0.95",
        "final_validation": final_validation,
    }
    (OUTPUT_ROOT / "selection_before_test.json").write_text(
        json.dumps(selection_record, indent=2)
    )
    final_test = evaluate(final_checkpoint, data_yaml, "test", "final_tuned_yolo11n_test")
    payload = {
        "hardware": torch.cuda.get_device_name(0),
        "seed": SEED,
        "trials": trials,
        "selected_learning_rate": best["learning_rate"],
        "final_training_duration_seconds": duration,
        "final_validation": final_validation,
        "final_test": final_test,
    }
    (OUTPUT_ROOT / "controlled_tuning_results.json").write_text(json.dumps(payload, indent=2))
    shutil.copy2(final_checkpoint, OUTPUT_ROOT / "yolo11n-rdd2022-tuned.pt")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

