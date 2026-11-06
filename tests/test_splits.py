from pathlib import Path

from roadguard.data.discovery import Sample
from roadguard.data.splits import create_grouped_split


def make_samples(country: str, count: int):
    return [
        Sample(f"{country}__{i}", country, Path(f"{i}.jpg"), Path(f"{i}.xml")) for i in range(count)
    ]


def test_grouped_split_is_deterministic_and_disjoint():
    samples = make_samples("Japan", 20) + make_samples("India", 20)
    first = create_grouped_split(samples, seed=42)
    second = create_grouped_split(samples, seed=42)
    assert first == second
    sets = {key: set(value) for key, value in first.items()}
    assert not sets["train"] & sets["val"]
    assert not sets["train"] & sets["test"]
    assert not sets["val"] & sets["test"]
    assert sum(map(len, sets.values())) == 40
    for split in sets.values():
        assert any(item.startswith("Japan__") for item in split)
        assert any(item.startswith("India__") for item in split)
