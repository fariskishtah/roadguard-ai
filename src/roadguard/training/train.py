from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
from ultralytics import YOLO

SUPPORTED_MODELS = ("yolov8n.pt", "yolo11n.pt", "yolo26n.pt")


def train_model(
    model_name: str,
    data_yaml: str | Path,
    output_root: str | Path,
    *,
    epochs: int = 35,
    image_size: int = 640,
    batch: int = 16,
    seed: int = 42,
    learning_rate: float = 0.01,
    weight_decay: float = 0.0005,
    device: str | int | None = None,
    run_name: str | None = None,
) -> Path:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Expected one of {SUPPORTED_MODELS}, got {model_name}")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = run_name or Path(model_name).stem
    started = time.time()
    model = YOLO(model_name)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        seed=seed,
        deterministic=True,
        optimizer="auto",
        lr0=learning_rate,
        weight_decay=weight_decay,
        patience=max(10, epochs // 3),
        cos_lr=True,
        close_mosaic=10,
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.35,
        degrees=2.0,
        translate=0.08,
        scale=0.35,
        shear=1.0,
        perspective=0.0002,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.8,
        mixup=0.0,
        project=str(output_root),
        name=run_name,
        exist_ok=True,
        device=device,
        plots=True,
        save=True,
        verbose=True,
    )
    run_dir = Path(results.save_dir)
    metadata = {
        "model": model_name,
        "architecture": Path(model_name).stem,
        "data_yaml": str(Path(data_yaml).resolve()),
        "epochs_requested": epochs,
        "image_size": image_size,
        "batch": batch,
        "seed": seed,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "device": str(device),
        "duration_seconds": time.time() - started,
        "torch_version": torch.__version__,
        "ultralytics_version": __import__("ultralytics").__version__,
        "platform": platform.platform(),
    }
    (run_dir / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return run_dir / "weights" / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="runs/train")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--device")
    parser.add_argument("--name")
    args = parser.parse_args()
    checkpoint = train_model(
        args.model,
        args.data,
        args.output,
        epochs=args.epochs,
        image_size=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        learning_rate=args.lr0,
        weight_decay=args.weight_decay,
        device=args.device,
        run_name=args.name,
    )
    print(checkpoint)


if __name__ == "__main__":
    main()
