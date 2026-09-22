import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_config():
    """A minimal but structurally complete config dict, independent of
    configs/config.yaml so these tests don't break if tuning values there
    change. test_config.py separately checks the real file loads."""
    return {
        "model": {
            "name": "yolo26n",
            "weights": "models/weights/yolo26n.pt",
            "device": "auto",
            "img_size": 640,
            "confidence": 0.35,
            "iou": 0.5,
            "half_precision": True,
        },
        "classes": {
            "vehicle_class_ids": [1, 2, 3, 5, 7],
            "names": {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"},
        },
        "tracking": {
            "tracker": "bytetrack",
            "persist": True,
            "track_high_thresh": 0.5,
            "track_low_thresh": 0.1,
            "new_track_thresh": 0.6,
            "track_buffer": 30,
            "match_thresh": 0.8,
        },
        "counting": {
            "line": {"point1": [0.0, 0.5], "point2": [1.0, 0.5], "margin_px": 15},
            "direction_labels": ["IN", "OUT"],
        },
        "io": {
            "video_path": "data/raw/sample.mp4",
            "output_video_path": "outputs/videos/processed.mp4",
            "output_results_path": "outputs/results/counts.json",
        },
    }


def make_fake_box(cls_id: int, conf: float, xyxy: tuple, track_id: int | None = None):
    """Mimics an Ultralytics Boxes row closely enough for our code: .cls/.conf
    are 1-element tensors read with .item(), .xyxy is a (1, 4) tensor read
    with [0].tolist(), .id is a 1-element tensor (or None, pre-tracking)."""
    box = SimpleNamespace()
    box.cls = torch.tensor([float(cls_id)])
    box.conf = torch.tensor([float(conf)])
    box.xyxy = torch.tensor([list(xyxy)], dtype=torch.float32)
    box.id = torch.tensor([float(track_id)]) if track_id is not None else None
    return box


class FakeBoxesList(list):
    """boxes.id is checked once for the whole batch (None means "tracker
    hasn't confirmed anything yet"), then boxes is iterated per-box."""

    @property
    def id(self):
        for box in self:
            if box.id is not None:
                return box.id
        return None


def make_fake_result(boxes: list):
    return SimpleNamespace(boxes=FakeBoxesList(boxes))


@pytest.fixture
def fake_box():
    return make_fake_box


@pytest.fixture
def make_fake_yolo(monkeypatch):
    """Patches ultralytics.YOLO in both detector.py and tracker.py so
    constructing a VehicleDetector/VehicleTracker never touches disk or a
    GPU. Call with the list of boxes each predict()/track() call should
    return (one list of boxes per call, consumed in order)."""

    def _install(module_path: str, results_per_call: list[list]):
        call_results = iter(results_per_call)

        class FakeYOLO:
            def __init__(self, *args, **kwargs):
                pass

            def predict(self, *args, **kwargs):
                self.last_call_kwargs = kwargs
                return [make_fake_result(next(call_results))]

            def track(self, *args, **kwargs):
                self.last_call_kwargs = kwargs
                return [make_fake_result(next(call_results))]

        monkeypatch.setattr(module_path, FakeYOLO)
        return FakeYOLO

    return _install
