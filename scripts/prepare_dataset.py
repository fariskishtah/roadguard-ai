from __future__ import annotations

import argparse
import json

from roadguard.data.prepare import prepare_yolo_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert RDD2022 Pascal VOC data to frozen YOLO splits"
    )
    parser.add_argument("raw_root")
    parser.add_argument("output_root")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-per-country", type=int)
    args = parser.parse_args()
    result = prepare_yolo_dataset(
        args.raw_root,
        args.output_root,
        seed=args.seed,
        limit_per_country=args.limit_per_country,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
