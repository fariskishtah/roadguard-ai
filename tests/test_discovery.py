from pathlib import Path

from roadguard.data.discovery import discover_training_samples


def write_pair(root: Path, country: str, stem: str):
    image = root / country / "train" / "images" / f"{stem}.jpg"
    xml = root / country / "train" / "annotations" / f"{stem}.xml"
    image.parent.mkdir(parents=True, exist_ok=True)
    xml.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"image")
    xml.write_text("<annotation />")


def test_duplicate_filenames_from_countries_do_not_overwrite(tmp_path):
    write_pair(tmp_path, "Japan", "000001")
    write_pair(tmp_path, "India", "000001")
    samples = discover_training_samples(tmp_path)
    assert len(samples) == 2
    assert {sample.country for sample in samples} == {"Japan", "India"}
    assert {sample.sample_id for sample in samples} == {"Japan__000001", "India__000001"}
