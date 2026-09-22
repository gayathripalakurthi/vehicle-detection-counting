from dataclasses import dataclass

from ultralytics import YOLO

from src.detection.detector import resolve_device


@dataclass
class TrackedObject:
    track_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


class VehicleTracker:
    """Wraps Ultralytics' built-in tracker (ByteTrack/BoT-SORT) to assign
    persistent IDs to vehicles across frames. Must be fed frames in order
    from a single video - internal tracker state is stateful."""

    def __init__(self, config: dict):
        model_cfg = config["model"]
        track_cfg = config["tracking"]

        self.device = resolve_device(model_cfg["device"])
        self.confidence = model_cfg["confidence"]
        self.iou = model_cfg["iou"]
        self.img_size = model_cfg["img_size"]
        self.vehicle_class_ids = list(config["classes"]["vehicle_class_ids"])
        self.class_names = config["classes"]["names"]

        tracker_name = track_cfg["tracker"]
        self.tracker_yaml = f"{tracker_name}.yaml"  # bytetrack.yaml / botsort.yaml ship with ultralytics

        self.model = YOLO(model_cfg["weights"])

    def track(self, frame) -> list[TrackedObject]:
        results = self.model.track(
            frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.img_size,
            device=self.device,
            classes=self.vehicle_class_ids,
            tracker=self.tracker_yaml,
            persist=True,
            verbose=False,
        )
        tracked = []
        boxes = results[0].boxes
        if boxes.id is None:
            return tracked  # tracker hasn't confirmed any tracks yet this frame

        for box in boxes:
            class_id = int(box.cls.item())
            tracked.append(
                TrackedObject(
                    track_id=int(box.id.item()),
                    xyxy=tuple(box.xyxy[0].tolist()),
                    confidence=float(box.conf.item()),
                    class_id=class_id,
                    class_name=self.class_names.get(class_id, str(class_id)),
                )
            )
        return tracked

    def reset(self):
        """Clear tracker state - call before starting a new video."""
        self.model.predictor = None
