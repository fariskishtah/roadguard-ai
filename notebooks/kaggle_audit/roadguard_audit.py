"""Comprehensive, image-aware audit of the exact RoadGuard 6,000-image subset."""

from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "imagehash>=4.3,<5"],
    check=True,
)

import imagehash  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402

SEED = 42
MAX_PER_COUNTRY = 1000
INPUT_BASE = Path("/kaggle/input")
OUTPUT_ROOT = Path("/kaggle/working/roadguard-audit-output")


def locate_input() -> Path:
    patterns = ("*/data.yaml", "datasets/*/*/data.yaml", "datasets/*/*/versions/*/data.yaml")
    for pattern in patterns:
        for path in INPUT_BASE.glob(pattern):
            if (path.parent / "train" / "images").is_dir():
                return path.parent
    raise FileNotFoundError("RDD2022 transport mirror was not mounted")


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


def discover_and_select(input_root: Path) -> tuple[list[dict], dict]:
    by_country = defaultdict(list)
    inventory = Counter()
    for source_split in ("train", "val"):
        image_dir = input_root / source_split / "images"
        label_dir = input_root / source_split / "labels"
        for image in sorted(image_dir.iterdir()):
            if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            country = country_from_stem(image.stem)
            inventory[country] += 1
            if country != "Unknown":
                by_country[country].append(
                    {
                        "country": country,
                        "stem": image.stem,
                        "image": image,
                        "label": label_dir / f"{image.stem}.txt",
                        "group": sequence_group(country, image.stem),
                    }
                )
    rng = random.Random(SEED)
    selected = []
    for country in sorted(by_country):
        candidates = sorted(by_country[country], key=lambda item: item["stem"])
        rng.shuffle(candidates)
        selected.extend(candidates[:MAX_PER_COUNTRY])
    return sorted(selected, key=lambda item: (item["country"], item["stem"])), dict(inventory)


class BKTree:
    def __init__(self) -> None:
        self.root: tuple[int, list[dict], dict] | None = None

    def add(self, value: int, record: dict) -> None:
        if self.root is None:
            self.root = (value, [record], {})
            return
        node = self.root
        while True:
            node_value, records, children = node
            distance = (node_value ^ value).bit_count()
            if distance == 0:
                records.append(record)
                return
            if distance not in children:
                children[distance] = (value, [record], {})
                return
            node = children[distance]

    def query(self, value: int, threshold: int) -> list[tuple[int, dict]]:
        if self.root is None:
            return []
        matches = []
        stack = [self.root]
        while stack:
            node_value, records, children = stack.pop()
            distance = (node_value ^ value).bit_count()
            if distance <= threshold:
                matches.extend((distance, record) for record in records)
            stack.extend(
                child
                for edge, child in children.items()
                if distance - threshold <= edge <= distance + threshold
            )
        return matches


def quantiles(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def parse_label(path: Path) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    if not path.is_file():
        return [], [f"{path}:missing"]
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
            x_center, y_center, width, height = (float(value) for value in parts[1:])
        except ValueError:
            errors.append(f"{path}:{line_number}:parse")
            continue
        if class_id not in range(4):
            errors.append(f"{path}:{line_number}:class_{class_id}")
            continue
        if not all(0 <= value <= 1 for value in (x_center, y_center, width, height)):
            errors.append(f"{path}:{line_number}:range")
            continue
        if width <= 0 or height <= 0:
            errors.append(f"{path}:{line_number}:non_positive_size")
            continue
        if x_center - width / 2 < 0 or x_center + width / 2 > 1:
            errors.append(f"{path}:{line_number}:x_bounds")
            continue
        if y_center - height / 2 < 0 or y_center + height / 2 > 1:
            errors.append(f"{path}:{line_number}:y_bounds")
            continue
        boxes.append((class_id, x_center, y_center, width, height))
    return boxes, errors


def save_histogram(values: list[float], title: str, x_label: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.4))
    axis.hist(values, bins=50, color="#1687ff", alpha=0.9)
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_xlabel(x_label)
    axis.set_ylabel("Annotations")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    selected, inventory = discover_and_select(locate_input())
    if len(selected) != 6000:
        raise RuntimeError(f"Expected 6,000 selected images, found {len(selected)}")
    missing_labels = []
    corrupt_images = []
    invalid_boxes = []
    resolutions = Counter()
    classes = Counter()
    annotations_per_image = []
    class_sets = []
    widths = []
    heights = []
    areas = []
    aspects = []
    pixel_widths = []
    pixel_heights = []
    exact_hashes = defaultdict(list)
    for index, record in enumerate(selected, start=1):
        boxes, errors = parse_label(record["label"])
        if not record["label"].is_file():
            missing_labels.append(str(record["label"]))
        invalid_boxes.extend(errors)
        try:
            with Image.open(record["image"]) as image:
                image.verify()
            with Image.open(record["image"]) as image:
                image = image.convert("RGB")
                record["width"], record["height"] = image.size
                record["phash"] = int(str(imagehash.phash(image)), 16)
        except (OSError, UnidentifiedImageError) as exc:
            corrupt_images.append(f"{record['image']}:{exc}")
            continue
        record["sha256"] = sha256_file(record["image"])
        exact_hashes[record["sha256"]].append(record)
        record["boxes"] = boxes
        resolutions[f"{record['width']}x{record['height']}"] += 1
        annotations_per_image.append(len(boxes))
        class_set = {box[0] for box in boxes}
        class_sets.append(class_set)
        classes.update(box[0] for box in boxes)
        for _, _, _, width, height in boxes:
            widths.append(width)
            heights.append(height)
            areas.append(width * height)
            aspects.append(width / height)
            pixel_widths.append(width * record["width"])
            pixel_heights.append(height * record["height"])
        if index % 500 == 0:
            print(f"Audited {index}/{len(selected)} images", flush=True)

    exact_duplicate_groups = [records for records in exact_hashes.values() if len(records) > 1]
    hash_leaders = {
        digest: min(record["group"] for record in records)
        for digest, records in exact_hashes.items()
    }
    for record in selected:
        if "sha256" in record:
            record["split"] = split_for_group(hash_leaders[record["sha256"]])

    tree = BKTree()
    near_duplicate_pairs = []
    near_duplicate_pair_count = 0
    cross_split_candidate_count = 0
    for record in selected:
        if "phash" not in record:
            continue
        for distance, other in tree.query(record["phash"], 4):
            if record["sha256"] == other["sha256"]:
                continue
            near_duplicate_pair_count += 1
            cross_split = record["split"] != other["split"]
            cross_split_candidate_count += int(cross_split)
            if len(near_duplicate_pairs) < 500:
                near_duplicate_pairs.append(
                    {
                        "first": f"{other['country']}__{other['stem']}",
                        "second": f"{record['country']}__{record['stem']}",
                        "phash_hamming_distance": distance,
                        "first_split": other["split"],
                        "second_split": record["split"],
                        "cross_split": cross_split,
                    }
                )
        tree.add(record["phash"], record)

    object_sizes = Counter()
    for area in areas:
        if area < 0.01:
            object_sizes["small_under_1pct"] += 1
        elif area < 0.05:
            object_sizes["medium_1_to_5pct"] += 1
        else:
            object_sizes["large_5pct_plus"] += 1
    audit = {
        "scope": "exact deterministic 6,000-image RoadGuard subset",
        "seed": SEED,
        "annotated_transport_inventory": sum(inventory.values()),
        "transport_images_by_country": inventory,
        "selected_images": len(selected),
        "selected_images_by_country": dict(Counter(record["country"] for record in selected)),
        "split_counts": dict(Counter(record.get("split") for record in selected)),
        "total_annotations": sum(classes.values()),
        "annotations_by_class_id": dict(classes),
        "empty_label_images": sum(not class_set for class_set in class_sets),
        "multi_damage_type_images": sum(len(class_set) > 1 for class_set in class_sets),
        "images_with_annotations": sum(bool(class_set) for class_set in class_sets),
        "annotations_per_image": quantiles(annotations_per_image),
        "resolution_counts": dict(resolutions),
        "normalized_box_width": quantiles(widths),
        "normalized_box_height": quantiles(heights),
        "normalized_box_area": quantiles(areas),
        "box_aspect_ratio_width_over_height": quantiles(aspects),
        "box_width_pixels": quantiles(pixel_widths),
        "box_height_pixels": quantiles(pixel_heights),
        "object_size_distribution_by_image_area": dict(object_sizes),
        "missing_annotation_files": missing_labels,
        "corrupted_images": corrupt_images,
        "invalid_bounding_boxes": invalid_boxes,
        "exact_duplicate_groups": [
            [f"{record['country']}__{record['stem']}" for record in records]
            for records in exact_duplicate_groups
        ],
        "exact_duplicate_cross_split_leakage": [],
        "near_duplicate_method": "64-bit perceptual hash; Hamming distance <= 4",
        "near_duplicate_interpretation": (
            "Candidates only, not definitive duplicates; visual confirmation is required."
        ),
        "near_duplicate_candidate_pairs": near_duplicate_pair_count,
        "near_duplicate_cross_split_candidate_pairs": cross_split_candidate_count,
        "near_duplicate_examples_capped_at_500": near_duplicate_pairs,
        "geographic_independence_claimed": False,
    }
    (OUTPUT_ROOT / "comprehensive_dataset_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    save_histogram(
        areas,
        "Bounding-box area distribution",
        "Normalized fraction of image area",
        OUTPUT_ROOT / "box_area_distribution.png",
    )
    save_histogram(
        aspects,
        "Bounding-box aspect-ratio distribution",
        "Width / height",
        OUTPUT_ROOT / "box_aspect_ratio_distribution.png",
    )
    save_histogram(
        annotations_per_image,
        "Annotations per selected image",
        "Annotation count",
        OUTPUT_ROOT / "annotations_per_image.png",
    )
    concise = {
        key: value
        for key, value in audit.items()
        if key != "near_duplicate_examples_capped_at_500"
    }
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
