"""Phase 8: run detection + ByteTrack tracking on a full video, draw
persistent track IDs and class labels, and save the annotated output.
No line-crossing counting yet - that's Phase 9."""

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tracking.tracker import VehicleTracker
from src.utils.config import load_config, resolve_path
from src.visualization.draw import draw_hud, draw_tracked_object


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None, help="video path (overrides config)")
    parser.add_argument("--output", default=None, help="output video path (overrides config)")
    parser.add_argument("--max-frames", type=int, default=None, help="limit frames processed (debug)")
    args = parser.parse_args()

    config = load_config()
    video_path = resolve_path(args.source or config["io"]["video_path"])
    output_path = resolve_path(args.output or config["io"]["output_video_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading tracker: {config['model']['name']} + {config['tracking']['tracker']}")
    tracker = VehicleTracker(config)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    seen_ids = {}  # class_name -> set of track_ids ever seen
    frame_idx = 0
    inference_times = []
    limit = args.max_frames or total_frames

    while frame_idx < limit:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.perf_counter()
        tracked_objects = tracker.track(frame)
        inference_times.append(time.perf_counter() - t0)

        for obj in tracked_objects:
            draw_tracked_object(frame, obj)
            seen_ids.setdefault(obj.class_name, set()).add(obj.track_id)

        current_fps = 1.0 / inference_times[-1] if inference_times[-1] > 0 else 0
        hud_lines = [f"Frame {frame_idx}/{total_frames}", f"FPS: {current_fps:.1f}"]
        for cls_name, ids in sorted(seen_ids.items()):
            hud_lines.append(f"{cls_name}: {len(ids)}")
        draw_hud(frame, hud_lines)

        writer.write(frame)

        if frame_idx % 30 == 0:
            print(f"frame {frame_idx}/{total_frames} | {len(tracked_objects)} active tracks")
        frame_idx += 1

    cap.release()
    writer.release()

    print(f"\nSaved annotated video to: {output_path}")
    print("\n--- Unique tracks per class (whole video) ---")
    total = 0
    for cls_name, ids in sorted(seen_ids.items(), key=lambda x: -len(x[1])):
        print(f"  {cls_name}: {len(ids)}")
        total += len(ids)
    print(f"  Total: {total}")
    avg_fps = 1.0 / (sum(inference_times) / len(inference_times)) if inference_times else 0
    print(f"\nAvg tracking FPS: {avg_fps:.1f} over {frame_idx} frames")


if __name__ == "__main__":
    main()
