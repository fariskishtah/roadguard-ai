from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    sample_id: str
    country: str
    image_path: Path
    annotation_path: Path


COUNTRY_PATTERNS = {
    "Japan": re.compile(r"japan", re.I),
    "India": re.compile(r"india", re.I),
    "Czech": re.compile(r"czech", re.I),
    "Norway": re.compile(r"norway", re.I),
    "United_States": re.compile(r"united[_ -]?states|usa", re.I),
    "China": re.compile(r"china", re.I),
}


def infer_country(path: Path) -> str:
    text = str(path)
    for country, pattern in COUNTRY_PATTERNS.items():
        if pattern.search(text):
            return country
    return "Unknown"


def discover_training_samples(root: str | Path) -> list[Sample]:
    root = Path(root)
    images: dict[str, list[Path]] = defaultdict(list)
    for pattern in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"):
        for path in root.rglob(pattern):
            if "train" in str(path).lower():
                images[path.stem].append(path)
    xmls = [p for p in root.rglob("*.xml") if "train" in str(p).lower()]
    samples: list[Sample] = []
    for xml in xmls:
        candidates = images.get(xml.stem, [])
        if not candidates:
            continue
        annotation_country = infer_country(xml)
        same_country = [path for path in candidates if infer_country(path) == annotation_country]
        image = sorted(same_country or candidates)[0]
        country = infer_country(image)
        sample_id = f"{country}__{image.stem}"
        samples.append(Sample(sample_id, country, image, xml))
    return sorted(samples, key=lambda item: item.sample_id)
