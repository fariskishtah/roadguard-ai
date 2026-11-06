from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from roadguard.constants import CODE_TO_ID
from roadguard.data.discovery import discover_training_samples
from roadguard.data.voc import parse_voc_xml


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(raw_root: str | Path, output_path: str | Path | None = None) -> dict:
    samples = discover_training_samples(raw_root)
    report: dict = {
        "images": len(samples),
        "annotations": 0,
        "images_by_country": Counter(),
        "annotations_by_class": Counter(),
        "multi_class_images": 0,
        "empty_images": 0,
        "corrupt_images": [],
        "dimension_mismatches": [],
        "invalid_boxes": [],
        "unknown_classes": Counter(),
        "exact_duplicate_groups": [],
        "resolutions": Counter(),
    }
    hashes: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        report["images_by_country"][sample.country] += 1
        try:
            with Image.open(sample.image_path) as image:
                image.verify()
            with Image.open(sample.image_path) as image:
                actual_size = image.size
        except (OSError, UnidentifiedImageError) as exc:
            report["corrupt_images"].append({"sample_id": sample.sample_id, "error": str(exc)})
            continue
        annotation = parse_voc_xml(sample.annotation_path)
        report["resolutions"][f"{actual_size[0]}x{actual_size[1]}"] += 1
        if actual_size != (annotation.width, annotation.height):
            report["dimension_mismatches"].append(sample.sample_id)
        seen_classes: set[str] = set()
        valid_count = 0
        for box in annotation.boxes:
            if box.class_code not in CODE_TO_ID:
                report["unknown_classes"][box.class_code] += 1
                continue
            seen_classes.add(box.class_code)
            valid_count += 1
            report["annotations_by_class"][box.class_code] += 1
            if (
                box.xmin < 0
                or box.ymin < 0
                or box.xmax > annotation.width
                or box.ymax > annotation.height
                or box.xmax <= box.xmin
                or box.ymax <= box.ymin
            ):
                report["invalid_boxes"].append(
                    {"sample_id": sample.sample_id, "class": box.class_code, "box": box.__dict__}
                )
        report["annotations"] += valid_count
        report["empty_images"] += int(valid_count == 0)
        report["multi_class_images"] += int(len(seen_classes) > 1)
        hashes[sha256_file(sample.image_path)].append(sample.sample_id)
    report["exact_duplicate_groups"] = [ids for ids in hashes.values() if len(ids) > 1]
    for key in (
        "images_by_country",
        "annotations_by_class",
        "unknown_classes",
        "resolutions",
    ):
        report[key] = dict(report[key])
    if output_path:
        Path(output_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
