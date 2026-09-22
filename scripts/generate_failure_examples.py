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

    model = YOLO(config["model"]["weights"])
    results = model.predict(frame, conf=0.01, iou=0.5, imgsz=640, device="0", verbose=False)
    r = results[0]

    annotated = frame.copy()
    for box in r.boxes:
        cid = int(box.cls.item())
        if cid not in (1, 2, 3, 5, 7):  # vehicle classes only
            continue
        conf = float(box.conf.item())
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        color = (0, 255, 0) if conf >= config["model"]["confidence"] else (0, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{r.names[cid]} {conf:.2f}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    out_path = out_dir / "camera_angle_nadir_view.jpg"
    cv2.imwrite(str(out_path), annotated)
    below_threshold = sum(
        1 for box in r.boxes
        if int(box.cls.item()) in (1, 2, 3, 5, 7) and float(box.conf.item()) < config["model"]["confidence"]
    )
    print(f"Saved {out_path} - {below_threshold} vehicle boxes found but below the "
          f"{config['model']['confidence']} confidence threshold (red = rejected, green = would pass)")


def generate_track_loss_example(config, out_dir: Path):
    video_path = resolve_path(config["io"]["video_path"])
    tracker = VehicleTracker(config)
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    counter = LineCounter(config, width, height)

    target_id = 54
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
        if last_seen_frame and frame_idx > last_seen_frame[0] + 40:
            break  # past the tracker's track_buffer (30 frames), so any gap this long is a real drop,
            # not just a frame or two of missed detection while ByteTrack still holds the ID

    if not last_seen_frame:
        print(f"Track {target_id} not found in this run (tracker IDs aren't guaranteed stable "
              "across runs/versions) - skipping track-loss example")
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

    cap.release()


def main():
    config = load_config()
    out_dir = resolve_path("outputs/plots/failure_cases")
    out_dir.mkdir(parents=True, exist_ok=True)

    generate_camera_angle_example(config, out_dir)
    generate_track_loss_example(config, out_dir)


if __name__ == "__main__":
    main()
