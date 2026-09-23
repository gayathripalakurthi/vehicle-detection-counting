from src.counting.line_counter import LineCounter
from src.tracking.tracker import TrackedObject


def make_obj(track_id, y_bottom, class_name="car", x_center=100.0):
    # anchor point is bottom-center of the box: ((x1+x2)/2, y2)
    xyxy = (x_center - 10, 0.0, x_center + 10, y_bottom)
    return TrackedObject(track_id=track_id, xyxy=xyxy, confidence=0.9, class_id=2, class_name=class_name)


def test_line_endpoints_convert_normalized_to_pixels(sample_config):
    counter = LineCounter(sample_config, frame_width=1000, frame_height=800)
    assert counter.p1 == (0.0, 400.0)
    assert counter.p2 == (1000.0, 400.0)


def test_crossing_counts_exactly_once_even_if_it_oscillates_back(sample_config):
    # line at y=400, margin=15
    counter = LineCounter(sample_config, frame_width=1000, frame_height=800)

    events = counter.update([make_obj(1, y_bottom=300)], frame_idx=0)  # above line
    assert events == []

    events = counter.update([make_obj(1, y_bottom=500)], frame_idx=1)  # crosses below
    assert len(events) == 1
    assert events[0].direction == "OUT"
    assert counter.total() == 1

    # moving back above the line must NOT count a second time
    events = counter.update([make_obj(1, y_bottom=300)], frame_idx=2)
    assert events == []
    assert counter.total() == 1


def test_jitter_within_margin_never_registers_a_crossing(sample_config):
    counter = LineCounter(sample_config, frame_width=1000, frame_height=800)  # margin_px=15

    # a vehicle idling right at the line, bouncing a few px either side - all within the 15px margin
    for frame_idx, y in enumerate([395, 405, 398, 402, 396, 404]):
        events = counter.update([make_obj(1, y_bottom=y)], frame_idx=frame_idx)
        assert events == []

    assert counter.total() == 0


def test_direction_label_depends_on_crossing_direction(sample_config):
    counter = LineCounter(sample_config, frame_width=1000, frame_height=800)

    counter.update([make_obj(1, y_bottom=300)], frame_idx=0)  # track 1 starts above
    counter.update([make_obj(2, y_bottom=500)], frame_idx=0)  # track 2 starts below

    events_a = counter.update([make_obj(1, y_bottom=500)], frame_idx=1)  # 1 crosses downward
    events_b = counter.update([make_obj(2, y_bottom=300)], frame_idx=1)  # 2 crosses upward

    assert events_a[0].direction == "OUT"
    assert events_b[0].direction == "IN"
    assert counter.totals_by_direction() == {"IN": 1, "OUT": 1}


def test_counts_aggregate_per_class(sample_config):
    counter = LineCounter(sample_config, frame_width=1000, frame_height=800)

    counter.update([make_obj(1, y_bottom=300, class_name="car"), make_obj(2, y_bottom=300, class_name="bus")], 0)
    counter.update([make_obj(1, y_bottom=500, class_name="car"), make_obj(2, y_bottom=500, class_name="bus")], 1)
    counter.update([make_obj(3, y_bottom=300, class_name="car")], 2)
    counter.update([make_obj(3, y_bottom=500, class_name="car")], 3)

    assert counter.counts["car"] == {"IN": 0, "OUT": 2}
    assert counter.counts["bus"] == {"IN": 0, "OUT": 1}
    assert counter.total() == 3


def test_is_counted_reflects_track_state(sample_config):
    counter = LineCounter(sample_config, frame_width=1000, frame_height=800)
    assert counter.is_counted(1) is False

    counter.update([make_obj(1, y_bottom=300)], 0)
    assert counter.is_counted(1) is False  # seen, but hasn't crossed yet

    counter.update([make_obj(1, y_bottom=500)], 1)
    assert counter.is_counted(1) is True


def test_confirmation_frames_ignores_a_single_frame_flip(sample_config):
    # simulates exactly what we saw in practice: a borderline-confidence box flips side
    # for one frame (e.g. GPU inference run-to-run variance) then reverts - with
    # confirmation_frames=3 this must NOT count as a crossing
    config = {**sample_config, "counting": {**sample_config["counting"],
              "line": {**sample_config["counting"]["line"], "confirmation_frames": 3}}}
    counter = LineCounter(config, frame_width=1000, frame_height=800)

    counter.update([make_obj(1, y_bottom=300)], 0)  # establish above the line
    events = counter.update([make_obj(1, y_bottom=500)], 1)  # one-frame flip below
    assert events == []
    events = counter.update([make_obj(1, y_bottom=300)], 2)  # reverts back above
    assert events == []
    assert counter.total() == 0
    assert counter.is_counted(1) is False


def test_confirmation_frames_counts_a_sustained_crossing(sample_config):
    config = {**sample_config, "counting": {**sample_config["counting"],
              "line": {**sample_config["counting"]["line"], "confirmation_frames": 3}}}
    counter = LineCounter(config, frame_width=1000, frame_height=800)

    counter.update([make_obj(1, y_bottom=300)], 0)  # above
    assert counter.update([make_obj(1, y_bottom=500)], 1) == []  # below, 1/3
    assert counter.update([make_obj(1, y_bottom=505)], 2) == []  # below, 2/3
    events = counter.update([make_obj(1, y_bottom=510)], 3)  # below, 3/3 - confirmed
    assert len(events) == 1
    assert counter.total() == 1

    # further frames on the same (now-confirmed) side must not double-count
    assert counter.update([make_obj(1, y_bottom=520)], 4) == []
    assert counter.total() == 1
