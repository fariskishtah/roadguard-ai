from __future__ import annotations

CLASS_CODES = ("D00", "D10", "D20", "D40")
CLASS_NAMES = (
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Pothole",
)
CODE_TO_ID = {code: idx for idx, code in enumerate(CLASS_CODES)}
ID_TO_CODE = {idx: code for code, idx in CODE_TO_ID.items()}
ID_TO_NAME = dict(enumerate(CLASS_NAMES))
CLASS_COLORS_RGB = {
    0: (0, 184, 255),
    1: (87, 108, 255),
    2: (255, 153, 51),
    3: (255, 82, 82),
}
