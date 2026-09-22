"""Generates annotated evidence images for known failure modes, for the
project's failure-case analysis. Not part of the main pipeline - a
one-off documentation aid, kept as a script rather than a notebook so
it's reproducible and diffable.

Failure mode 1 - camera angle (nadir/top-down drone view): a
COCO-pretrained detector is trained almost entirely on street/eye-level
imagery, so straight-down views push detection confidence far below any
usable threshold even though the model "sees" the right shape.

Failure mode 2 - track loss near the counting line: a vehicle at
borderline detection confidence gets tracked steadily for 80+ frames,
then the track drops a few pixels short of the counting line, causing
a genuine missed count (found via scripts/evaluate.py's recall check).
"""

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.counting.line_counter import LineCounter
from src.evaluation.recall_check import RecallChecker
from src.tracking.tracker import VehicleTracker
from src.utils.config import load_config, resolve_path
from ultralytics import YOLO


def generate_camera_angle_example(config, out_dir: Path):
    video_path = resolve_path("data/raw/sample_traffic.mp4")  # the nadir drone clip
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"Could not read from {video_path}, skipping camera-angle example")
        return

    vehicle_class_ids = set(config["classes"]["vehicle_class_ids"])  # model's own id scheme, not hardcoded COCO ids

    model = YOLO(config["model"]["weights"])
    results = model.predict(frame, conf=0.01, iou=0.5, imgsz=640, device="0", verbose=False)
    r = results[0]

    annotated = frame.copy()
    for box in r.boxes:
        cid = int(box.cls.item())
        if cid not in vehicle_class_ids:
            continue
        conf = float(box.conf.item())
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        color = (0, 255, 0) if conf >= config["model"]["confidence"] else (0, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{r.names[cid]} {conf:.2f}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    out_path = out_dir / "camera_angle_nadir_view.jpg"
    cv2.imwrite(str(out_path), annotated)
    vehicle_boxes = [box for box in r.boxes if int(box.cls.item()) in vehicle_class_ids]
    below_threshold = sum(1 for box in vehicle_boxes if float(box.conf.item()) < config["model"]["confidence"])
    print(f"Saved {out_path} - {len(vehicle_boxes)} vehicle boxes found, {below_threshold} below the "
          f"{config['model']['confidence']} confidence threshold (red = rejected, green = would pass)")


def _run_full_pass(config, video_path):
    """One full tracker+counter pass over the video. Returns (recall_checker, counter)."""
    tracker = VehicleTracker(config)
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    counter = LineCounter(config, width, height)
    recall_checker = RecallChecker(threshold_px=120)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        tracked = tracker.track(frame)
        counter.update(tracked, frame_idx)
        recall_checker.observe(tracked, counter)
        frame_idx += 1
    cap.release()
    return recall_checker, counter


def generate_track_loss_example(config, out_dir: Path):
    """Finds a near-miss dynamically (via the same recall-check logic
    scripts/evaluate.py uses) rather than hardcoding a track id - which
    model is active, and its confidence/tracking behavior, changes which
    track (if any) ends up being the closest near-miss.

    Two passes over the video rather than one: the target track id isn't
    known until the whole video's been seen, and keeping every 4K frame
    in memory to avoid a second pass would need tens of GB of RAM."""
    video_path = resolve_path(config["io"]["video_path"])

    recall_checker, counter = _run_full_pass(config, video_path)
    near_misses = recall_checker.near_misses(counter)
    if not near_misses:
        print("No near-miss tracks found in this run - skipping track-loss example "
              "(not necessarily a problem: it just means every track that got close to "
              "the line was either counted or stayed far from it)")
        return
    target_id = near_misses[0].track_id  # closest to the line without being counted

    tracker = VehicleTracker(config)
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    counter = LineCounter(config, width, height)  # fresh instance - tracker state doesn't carry over

    last_seen_frame = None
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        tracked = tracker.track(frame)
        counter.update(tracked, frame_idx)
        for obj in tracked:
            if obj.track_id == target_id:
                last_seen_frame = (frame_idx, frame.copy(), obj)
        frame_idx += 1
    cap.release()

    if not last_seen_frame:
        print(f"Track {target_id} (identified as a near-miss in the first pass) wasn't found "
              "in the second pass - tracker isn't deterministic run-to-run, skipping")
        return

    idx, frame, obj = last_seen_frame
    x1, y1, x2, y2 = (int(v) for v in obj.xyxy)
    dist = counter.distance_to_line(obj.xyxy)
    annotated = frame.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(annotated, f"ID {obj.track_id} conf={obj.confidence:.2f} {abs(dist):.0f}px from line",
                (x1, max(0, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.line(annotated, (int(counter.p1[0]), int(counter.p1[1])), (int(counter.p2[0]), int(counter.p2[1])),
              (0, 220, 255), 3)

    out_path = out_dir / "track_loss_near_line.jpg"
    cv2.imwrite(str(out_path), annotated)
    print(f"Saved {out_path} - last confirmed sighting of track {target_id} at frame {idx}, "
          f"{abs(dist):.0f}px short of the line, confidence {obj.confidence:.2f}")


def main():
    config = load_config()
    out_dir = resolve_path("outputs/plots/failure_cases")
    out_dir.mkdir(parents=True, exist_ok=True)

    generate_camera_angle_example(config, out_dir)
    generate_track_loss_example(config, out_dir)


if __name__ == "__main__":
    main()
