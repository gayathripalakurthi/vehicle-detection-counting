"""Phase 9: full pipeline - detect, track, count vehicles crossing a
configured line, save an annotated video plus a JSON results file."""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.counting.line_counter import LineCounter
from src.tracking.tracker import VehicleTracker
from src.utils.config import load_config, resolve_path
from src.visualization.draw import draw_counting_line, draw_hud, draw_tracked_object


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None, help="video path (overrides config)")
    parser.add_argument("--output", default=None, help="output video path (overrides config)")
    parser.add_argument("--results", default=None, help="output JSON results path (overrides config)")
    parser.add_argument("--max-frames", type=int, default=None, help="limit frames processed (debug)")
    args = parser.parse_args()

    config = load_config()
    video_path = resolve_path(args.source or config["io"]["video_path"])
    output_video_path = resolve_path(args.output or config["io"]["output_video_path"])
    output_results_path = resolve_path(args.results or config["io"]["output_results_path"])
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    output_results_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading tracker: {config['model']['name']} + {config['tracking']['tracker']}")
    tracker = VehicleTracker(config)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    counter = LineCounter(config, width, height)
    writer = cv2.VideoWriter(str(output_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_idx = 0
    inference_times = []
    limit = args.max_frames or total_frames

    while frame_idx < limit:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.perf_counter()
        tracked_objects = tracker.track(frame)
        events = counter.update(tracked_objects, frame_idx)
        inference_times.append(time.perf_counter() - t0)

        for obj in tracked_objects:
            draw_tracked_object(frame, obj)
        draw_counting_line(frame, counter.p1, counter.p2)

        current_fps = 1.0 / inference_times[-1] if inference_times[-1] > 0 else 0
        totals = counter.totals_by_direction()
        hud_lines = [
            f"Frame {frame_idx}/{total_frames}  FPS: {current_fps:.1f}",
            f"{counter.in_label}: {totals[counter.in_label]}   {counter.out_label}: {totals[counter.out_label]}",
            f"Total: {counter.total()}",
        ]
        for cls_name, d in sorted(counter.counts.items()):
            hud_lines.append(f"{cls_name}: {sum(d.values())}")
        draw_hud(frame, hud_lines)

        writer.write(frame)

        for e in events:
            print(f"frame {e.frame_idx}: track {e.track_id} ({e.class_name}) crossed -> {e.direction}")

        frame_idx += 1

    cap.release()
    writer.release()

    avg_fps = 1.0 / (sum(inference_times) / len(inference_times)) if inference_times else 0
    results = {
        "video": str(video_path),
        "frames_processed": frame_idx,
        "avg_processing_fps": round(avg_fps, 2),
        "counts_by_class": counter.counts,
        "totals_by_direction": counter.totals_by_direction(),
        "total_count": counter.total(),
    }
    with open(output_results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved annotated video to: {output_video_path}")
    print(f"Saved results JSON to: {output_results_path}")
    print("\n--- Counting summary ---")
    for cls_name, d in sorted(counter.counts.items(), key=lambda x: -sum(x[1].values())):
        print(f"  {cls_name}: {d}")
    print(f"  Totals by direction: {results['totals_by_direction']}")
    print(f"  Total count: {results['total_count']}")
    print(f"\nAvg processing FPS: {avg_fps:.1f} over {frame_idx} frames")


if __name__ == "__main__":
    main()
