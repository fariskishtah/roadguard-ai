from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from roadguard.constants import CLASS_CODES, CLASS_NAMES


def save_bar(frame: pd.DataFrame, x: str, y: str, title: str, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 5.4))
    palette = sns.color_palette("Blues", n_colors=max(3, frame[x].nunique()))
    sns.barplot(data=frame, x=x, y=y, hue=x, palette=palette, legend=False, ax=axis)
    axis.set_title(title, loc="left", fontweight="bold", fontsize=15)
    axis.grid(axis="y", alpha=0.2)
    sns.despine()
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_dataset_audit(audit_path: Path, output_dir: Path) -> None:
    audit = json.loads(audit_path.read_text())
    countries = audit.get("transport_images_by_country") or audit.get("images_by_country")
    if countries:
        frame = pd.DataFrame(
            [{"country": country, "images": count} for country, count in countries.items()]
        )
        save_bar(
            frame,
            "country",
            "images",
            "RDD2022 image inventory by country",
            output_dir / "images_by_country.png",
        )
    classes = audit.get("selected_annotations_by_class_id") or audit.get("annotations_by_class_id")
    if classes:
        rows = []
        for key, count in classes.items():
            class_id = int(key)
            rows.append(
                {
                    "class": f"{CLASS_CODES[class_id]}\n{CLASS_NAMES[class_id]}",
                    "annotations": count,
                }
            )
        save_bar(
            pd.DataFrame(rows),
            "class",
            "annotations",
            "Selected-subset annotation distribution",
            output_dir / "class_distribution.png",
        )


def render_model_comparison(metrics_path: Path, output_dir: Path) -> None:
    frame = pd.DataFrame(json.loads(metrics_path.read_text()))
    metric_frame = frame.melt(
        id_vars="model",
        value_vars=["map50", "map50_95", "precision", "recall"],
        var_name="metric",
        value_name="value",
    )
    fig, axis = plt.subplots(figsize=(10, 5.8))
    sns.barplot(data=metric_frame, x="model", y="value", hue="metric", ax=axis)
    axis.set_ylim(0, 1)
    axis.set_title(
        "RoadGuard detector quality on the frozen test split", loc="left", fontweight="bold"
    )
    axis.grid(axis="y", alpha=0.2)
    sns.despine()
    fig.tight_layout()
    fig.savefig(output_dir / "model_quality_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    save_bar(
        frame,
        "model",
        "fps",
        "Model-only throughput (same Kaggle GPU)",
        output_dir / "fps_comparison.png",
    )
    save_bar(frame, "model", "size_mb", "Checkpoint size", output_dir / "model_size_comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=Path("artifacts/metrics/dataset_audit.json"))
    parser.add_argument(
        "--comparison", type=Path, default=Path("artifacts/metrics/model_comparison.json")
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.audit.exists():
        render_dataset_audit(args.audit, args.output)
    if args.comparison.exists():
        render_model_comparison(args.comparison, args.output)


if __name__ == "__main__":
    main()
