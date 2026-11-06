from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import yaml

from roadguard.constants import CLASS_NAMES
from roadguard.data.discovery import Sample, discover_training_samples
from roadguard.data.splits import create_grouped_split, save_manifest
from roadguard.data.voc import annotation_to_yolo_lines, parse_voc_xml


def prepare_yolo_dataset(
    raw_root: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
    limit_per_country: int | None = None,
) -> dict:
    raw_root, output_root = Path(raw_root), Path(output_root)
    samples = discover_training_samples(raw_root)
    if limit_per_country:
        selected: list[Sample] = []
        counts: Counter[str] = Counter()
        for sample in samples:
            if counts[sample.country] < limit_per_country:
                selected.append(sample)
                counts[sample.country] += 1
        samples = selected
    if not samples:
        raise FileNotFoundError(f"No RDD2022 training samples found under {raw_root}")
    manifest = create_grouped_split(samples, seed=seed)
    split_by_id = {sample_id: split for split, ids in manifest.items() for sample_id in ids}
    stats = {"images": 0, "annotations": 0, "by_country": Counter(), "by_class": Counter()}
    for sample in samples:
        split = split_by_id[sample.sample_id]
        image_dir = output_root / "images" / split
        label_dir = output_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        target_stem = sample.sample_id
        target_image = image_dir / f"{target_stem}{sample.image_path.suffix.lower()}"
        if not target_image.exists():
            shutil.copy2(sample.image_path, target_image)
        annotation = parse_voc_xml(sample.annotation_path)
        lines = annotation_to_yolo_lines(annotation)
        (label_dir / f"{target_stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        stats["images"] += 1
        stats["annotations"] += len(lines)
        stats["by_country"][sample.country] += 1
        for line in lines:
            stats["by_class"][int(line.split()[0])] += 1
    save_manifest(manifest, output_root / "split_manifest.json", seed=seed)
    data_yaml = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(CLASS_NAMES)),
    }
    (output_root / "dataset.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    serializable = {
        "images": stats["images"],
        "annotations": stats["annotations"],
        "by_country": dict(stats["by_country"]),
        "by_class_id": {str(k): v for k, v in stats["by_class"].items()},
        "splits": {key: len(value) for key, value in manifest.items()},
        "seed": seed,
    }
    (output_root / "dataset_stats.json").write_text(json.dumps(serializable, indent=2) + "\n")
    return serializable
