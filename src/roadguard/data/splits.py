from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from roadguard.data.discovery import Sample


def create_grouped_split(
    samples: list[Sample],
    *,
    seed: int = 42,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> dict[str, list[str]]:
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("Split fractions must be between zero and one")
    if val_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than one")
    rng = random.Random(seed)
    by_country: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        by_country[sample.country].append(sample.sample_id)
    manifest = {"train": [], "val": [], "test": []}
    for ids in by_country.values():
        ids = sorted(ids)
        rng.shuffle(ids)
        n_test = round(len(ids) * test_fraction)
        n_val = round(len(ids) * val_fraction)
        manifest["test"].extend(ids[:n_test])
        manifest["val"].extend(ids[n_test : n_test + n_val])
        manifest["train"].extend(ids[n_test + n_val :])
    for split in manifest:
        manifest[split].sort()
    return manifest


def save_manifest(manifest: dict[str, list[str]], path: str | Path, seed: int = 42) -> None:
    payload = {"seed": seed, "strategy": "country-stratified deterministic split", **manifest}
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
