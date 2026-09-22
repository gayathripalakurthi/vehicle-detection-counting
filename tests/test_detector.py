import pytest
import torch

from src.detection.detector import VehicleDetector, resolve_device


def test_resolve_device_auto_prefers_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "0"


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"


def test_resolve_device_passes_explicit_values_through():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("0") == "0"


def test_detect_converts_boxes_and_filters_to_configured_classes(sample_config, make_fake_yolo, fake_box):
    # class 2 = car (in vehicle_class_ids), class 0 = person (not a vehicle class,
    # but Ultralytics itself does the classes= filtering - we just verify the
    # detector requests the right classes and converts what comes back)
    boxes = [
        fake_box(cls_id=2, conf=0.8, xyxy=(10, 20, 110, 220)),
        fake_box(cls_id=7, conf=0.6, xyxy=(50, 60, 150, 260)),
    ]
    make_fake_yolo("src.detection.detector.YOLO", results_per_call=[boxes])

    detector = VehicleDetector(sample_config)
    detections = detector.detect(frame="fake-frame-object")

    assert detector.model.last_call_kwargs["classes"] == sample_config["classes"]["vehicle_class_ids"]

    assert len(detections) == 2
    assert detections[0].class_id == 2
    assert detections[0].class_name == "car"
    assert detections[0].confidence == pytest.approx(0.8)
    assert detections[0].xyxy == (10.0, 20.0, 110.0, 220.0)
    assert detections[1].class_name == "truck"
