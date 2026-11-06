from pathlib import Path

import pytest

from roadguard.data.voc import VocBox, annotation_to_yolo_lines, parse_voc_xml, voc_box_to_yolo


def test_voc_box_to_yolo_normalizes_and_maps_class():
    result = voc_box_to_yolo(VocBox("D40", 10, 20, 50, 60), width=100, height=100)
    assert result == pytest.approx((3, 0.3, 0.4, 0.4, 0.4))


def test_voc_box_to_yolo_clips_to_image():
    result = voc_box_to_yolo(VocBox("D00", -5, 10, 120, 90), width=100, height=100)
    assert result == pytest.approx((0, 0.5, 0.5, 1.0, 0.8))


def test_voc_box_rejects_unknown_class():
    with pytest.raises(ValueError, match="Unsupported"):
        voc_box_to_yolo(VocBox("D99", 1, 1, 2, 2), 10, 10)


def test_parse_and_convert_voc(tmp_path: Path):
    xml = tmp_path / "sample.xml"
    xml.write_text(
        """<annotation><filename>x.jpg</filename><size><width>200</width><height>100</height></size>
        <object><name>D10</name><bndbox><xmin>20</xmin><ymin>10</ymin><xmax>60</xmax><ymax>30</ymax></bndbox></object>
        <object><name>D50</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>2</xmax><ymax>2</ymax></bndbox></object>
        </annotation>"""
    )
    annotation = parse_voc_xml(xml)
    assert annotation.filename == "x.jpg"
    assert annotation_to_yolo_lines(annotation) == ["1 0.20000000 0.20000000 0.20000000 0.20000000"]
