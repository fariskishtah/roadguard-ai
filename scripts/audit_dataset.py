from __future__ import annotations

import argparse
import json

from roadguard.data.audit import audit_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit extracted RDD2022 training data")
    parser.add_argument("raw_root")
    parser.add_argument("--output", default="artifacts/metrics/dataset_audit.json")
    args = parser.parse_args()
    report = audit_dataset(args.raw_root, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
