from dataclasses import dataclass

import torch
from ultralytics import YOLO


def resolve_device(device: str) -> str:
    if device == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    return device


@dataclass
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


class VehicleDetector:
    """Thin wrapper around an Ultralytics YOLO model, filtered to vehicle classes."""

    def __init__(self, config: dict):
        model_cfg = config["model"]
        self.device = resolve_device(model_cfg["device"])
        self.confidence = model_cfg["confidence"]
        self.iou = model_cfg["iou"]
        self.img_size = model_cfg["img_size"]
        self.quantize = 16 if (model_cfg.get("half_precision", False) and self.device != "cpu") else None
        self.vehicle_class_ids = set(config["classes"]["vehicle_class_ids"])
        self.class_names = config["classes"]["names"]

        self.model = YOLO(model_cfg["weights"])

    def detect(self, frame) -> list[Detection]:
        results = self.model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.img_size,
            device=self.device,
            quantize=self.quantize,
            classes=list(self.vehicle_class_ids),
            verbose=False,
        )
        detections = []
        for box in results[0].boxes:
            class_id = int(box.cls.item())
            detections.append(
                Detection(
                    xyxy=tuple(box.xyxy[0].tolist()),
                    confidence=float(box.conf.item()),
                    class_id=class_id,
                    class_name=self.class_names.get(class_id, str(class_id)),
                )
            )
        return detections
