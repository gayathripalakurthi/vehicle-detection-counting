"""Phase 15: head-to-head comparison of the COCO-pretrained baseline vs.
the VisDrone-fine-tuned model.

Two different kinds of metric, deliberately:

1. Precision/recall/F1 @ a fixed confidence threshold (our own IoU-matching
   evaluator, src/evaluation/detection_metrics.py), matched by class NAME.
   This is what lets the two models be compared fairly despite using
   different class-id numbering schemes (baseline: COCO ids like 2=car;
   fine-tuned: our own 0=bicycle..4=truck) - "at the threshold we actually
   run in production, how many TPs/FPs/FNs does each one produce".

2. Official mAP50/mAP50-95 for the fine-tuned model only, via Ultralytics'
   own validator (model.val()). The baseline's raw output space is COCO's
   80 classes, so running Ultralytics' val() on it against our 5-class
   VisDrone labels would silently misalign class indices and produce
   meaningless numbers - that's exactly the trap metric #1 is designed to
   avoid, by matching on class name instead.

Both models are also timed for FPS/latency on the same images.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.detector import VehicleDetector
from src.evaluation.detection_metrics import DetectionEvaluator
from src.utils.config import load_config, resolve_path

OUR_CLASS_NAMES = {0: "bicycle", 1: "car", 2: "motorcycle", 3: "bus", 4: "truck"}


def load_ground_truth(label_path: Path, img_w: int, img_h: int) -> list[tuple[str, tuple[float, float, float, float]]]:
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        if not line:
            continue
        cls, cx, cy, w, h = line.split()
        cls = int(cls)
        cx, cy, w, h = float(cx) * img_w, float(cy) * img_h, float(w) * img_w, float(h) * img_h
        xyxy = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        boxes.append((OUR_CLASS_NAMES[cls], xyxy))
    return boxes


def evaluate_model(detector: VehicleDetector, images: list[Path], labels_dir: Path, iou_threshold: float) -> dict:
    evaluator = DetectionEvaluator(iou_threshold=iou_threshold)
    inference_times = []

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]

        t0 = time.perf_counter()
        detections = detector.detect(frame)
        inference_times.append(time.perf_counter() - t0)

        predictions = [(d.class_name, d.confidence, d.xyxy) for d in detections]
        ground_truths = load_ground_truth(labels_dir / (img_path.stem + ".txt"), w, h)
        evaluator.update(predictions, ground_truths)

    fps_values = [1.0 / t for t in inference_times if t > 0]
    return {
        "detection_metrics": evaluator.report(),
        "performance": {
            "avg_fps": round(sum(fps_values) / len(fps_values), 2) if fps_values else 0,
            "avg_latency_ms": round(sum(inference_times) / len(inference_times) * 1000, 2) if inference_times else 0,
            "images_evaluated": len(inference_times),
        },
    }


def official_map_for_finetuned(finetuned_weights: Path, data_yaml: Path, device: str) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(finetuned_weights))
    metrics = model.val(data=str(data_yaml), split="val", device=device, verbose=False)
    return {
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50-95": round(float(metrics.box.map), 4),
        "per_class_mAP50-95": {
            OUR_CLASS_NAMES[i]: round(float(v), 4) for i, v in enumerate(metrics.box.maps)
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="cap number of val images evaluated (debug)")
    args = parser.parse_args()

    config = load_config()
    finetuned_weights = resolve_path("training/runs/visdrone_finetune/weights/best.pt")
    if not finetuned_weights.exists():
        raise FileNotFoundError(
            f"No fine-tuned weights at {finetuned_weights} - run training/train.py first."
        )

    dataset_yaml = yaml.safe_load(resolve_path("training/dataset.yaml").read_text())
    dataset_root = Path(dataset_yaml["path"])
    val_images_dir = dataset_root / "images" / "val"
    val_labels_dir = dataset_root / "labels" / "val"
    val_images = sorted(val_images_dir.glob("*.jpg"))
    if args.limit:
        val_images = val_images[: args.limit]

    print(f"Evaluating on {len(val_images)} VisDrone val images at IoU>=0.5, conf>={config['model']['confidence']}\n")

    baseline_config = {**config, "model": {**config["model"], "weights": "models/weights/yolo26n.pt"}}
    finetuned_config = {
        **config,
        "model": {**config["model"], "weights": str(finetuned_weights)},
        "classes": {"vehicle_class_ids": list(OUR_CLASS_NAMES.keys()), "names": OUR_CLASS_NAMES},
    }

    print("Running baseline (COCO-pretrained) model...")
    baseline_detector = VehicleDetector(baseline_config)
    baseline_results = evaluate_model(baseline_detector, val_images, val_labels_dir, iou_threshold=0.5)

    print("Running fine-tuned model...")
    finetuned_detector = VehicleDetector(finetuned_config)
    finetuned_results = evaluate_model(finetuned_detector, val_images, val_labels_dir, iou_threshold=0.5)

    print("Computing official mAP50/mAP50-95 for fine-tuned model...")
    from src.detection.detector import resolve_device
    finetuned_results["official_map"] = official_map_for_finetuned(
        finetuned_weights, resolve_path("training/dataset.yaml"), resolve_device(config["model"]["device"])
    )

    report = {"baseline": baseline_results, "finetuned": finetuned_results}
    out_path = resolve_path("outputs/results/model_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'Metric':<22}{'Baseline':>14}{'Fine-tuned':>14}")
    print("-" * 50)
    b_overall = baseline_results["detection_metrics"]["_overall"]
    f_overall = finetuned_results["detection_metrics"]["_overall"]
    for key in ("precision", "recall", "f1"):
        print(f"{key:<22}{b_overall[key]:>14}{f_overall[key]:>14}")
    print(f"{'avg FPS':<22}{baseline_results['performance']['avg_fps']:>14}{finetuned_results['performance']['avg_fps']:>14}")
    print(f"{'avg latency (ms)':<22}{baseline_results['performance']['avg_latency_ms']:>14}{finetuned_results['performance']['avg_latency_ms']:>14}")
    print(f"\nFine-tuned official mAP50: {finetuned_results['official_map']['mAP50']}")
    print(f"Fine-tuned official mAP50-95: {finetuned_results['official_map']['mAP50-95']}")

    print("\nPer-class precision/recall/F1:")
    for cls_name in OUR_CLASS_NAMES.values():
        b = baseline_results["detection_metrics"].get(cls_name, {"precision": 0, "recall": 0, "f1": 0})
        f = finetuned_results["detection_metrics"].get(cls_name, {"precision": 0, "recall": 0, "f1": 0})
        print(f"  {cls_name}: baseline P/R/F1={b['precision']}/{b['recall']}/{b['f1']}  "
              f"finetuned P/R/F1={f['precision']}/{f['recall']}/{f['f1']}")

    print(f"\nSaved full report to: {out_path}")


if __name__ == "__main__":
    main()
