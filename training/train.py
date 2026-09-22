"""Phase 14: fine-tune the pretrained YOLO26n detector on VisDrone,
starting from the same COCO-pretrained weights the baseline uses
(transfer learning, not training from scratch - our 5 classes are a
subset of what the model already partially knows, so this should
converge much faster than random-init training)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO

from src.detection.detector import resolve_device
from src.utils.config import load_config, resolve_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    train_cfg = config["training"]

    model = YOLO(config["model"]["weights"])
    device = resolve_device(config["model"]["device"])

    model.train(
        data=str(resolve_path(train_cfg["data_yaml"])),
        epochs=args.epochs or train_cfg["epochs"],
        batch=args.batch or train_cfg["batch_size"],
        imgsz=args.imgsz or train_cfg["img_size"],
        patience=train_cfg["patience"],
        optimizer=train_cfg["optimizer"],
        lr0=train_cfg["lr0"],
        device=device,
        seed=config["runtime"]["seed"],
        project=str(resolve_path("training/runs")),
        name="visdrone_finetune",
        # Explicit rather than relying on defaults, since we want the full
        # training record kept: per-epoch metrics (results.csv), loss/mAP
        # curves, confusion matrix, PR curves, and sample train/val batch
        # images all get written to training/runs/visdrone_finetune/.
        save=True,
        plots=True,
        verbose=True,
        # save_period keeps a checkpoint every 5 epochs in addition to
        # last.pt/best.pt, so a crash or a "which epoch was this?" question
        # later doesn't lose intermediate states.
        save_period=5,
        exist_ok=False,  # never overwrite a previous run's saved data
    )


if __name__ == "__main__":
    main()
