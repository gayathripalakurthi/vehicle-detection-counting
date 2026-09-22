from src.tracking.tracker import VehicleTracker


def test_track_returns_empty_before_tracker_confirms_any_ids(sample_config, make_fake_yolo, fake_box):
    # boxes.id is None for the first frame or two while ByteTrack warms up
    boxes = [fake_box(cls_id=2, conf=0.7, xyxy=(0, 0, 10, 10), track_id=None)]
    make_fake_yolo("src.tracking.tracker.YOLO", results_per_call=[boxes])

    tracker = VehicleTracker(sample_config)
    result = tracker.track(frame="fake-frame")

    assert result == []


def test_track_assigns_persistent_ids_across_frames(sample_config, make_fake_yolo, fake_box):
    frame1 = [fake_box(cls_id=2, conf=0.7, xyxy=(0, 0, 10, 10), track_id=5)]
    frame2 = [fake_box(cls_id=2, conf=0.75, xyxy=(2, 2, 12, 12), track_id=5)]
    make_fake_yolo("src.tracking.tracker.YOLO", results_per_call=[frame1, frame2])

    tracker = VehicleTracker(sample_config)
    result1 = tracker.track(frame="fake-frame-1")
    result2 = tracker.track(frame="fake-frame-2")

    assert result1[0].track_id == 5
    assert result2[0].track_id == 5
    assert result1[0].class_name == "car"


def test_track_passes_configured_tracker_yaml(sample_config, make_fake_yolo):
    make_fake_yolo("src.tracking.tracker.YOLO", results_per_call=[[]])
    tracker = VehicleTracker(sample_config)
    tracker.track(frame="fake-frame")
    assert tracker.model.last_call_kwargs["tracker"] == "bytetrack.yaml"
    assert tracker.model.last_call_kwargs["persist"] is True
