"""Phase 11 (partial): runtime performance stats + a recall sanity-check
that flags tracks which approached the counting line but were never
confirmed crossing it. Detection-quality metrics (mAP, precision,
recall against ground-truth boxes) need a labeled dataset and are
deferred to Stage B (Phase 12+) once one is selected."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.counting.line_counter import LineCounter
from src.evaluation.recall_check import RecallChecker
from src.tracking.tracker import VehicleTracker
from src.utils.config import load_config, resolve_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None, help="video path (overrides config)")
    parser.add_argument("--results", default=None, help="output JSON path (overrides outputs/results/evaluation.json)")
    parser.add_argument("--near-miss-threshold-px", type=float, default=120)
    args = parser.parse_args()

    config = load_config()
    video_path = resolve_path(args.source or config["io"]["video_path"])
    output_path = resolve_path(args.results or "outputs/results/evaluation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tracker = VehicleTracker(config)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    counter = LineCounter(config, width, height)
    recall_checker = RecallChecker(threshold_px=args.near_miss_threshold_px)

    frame_idx = 0
    frame_times = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        tracked = tracker.track(frame)
        counter.update(tracked, frame_idx)
        frame_times.append(time.perf_counter() - t0)
        recall_checker.observe(tracked, counter)
        frame_idx += 1

    cap.release()

    fps_values = [1.0 / t for t in frame_times if t > 0]
    near_misses = recall_checker.near_misses(counter)

    report = {
        "video": str(video_path),
        "frames_processed": frame_idx,
        "performance": {
            "avg_fps": round(statistics.mean(fps_values), 2) if fps_values else 0,
            "median_fps": round(statistics.median(fps_values), 2) if fps_values else 0,
            "p10_fps": round(sorted(fps_values)[int(len(fps_values) * 0.1)], 2) if fps_values else 0,
            "avg_latency_ms": round(statistics.mean(frame_times) * 1000, 2) if frame_times else 0,
        },
        "counting": {
            "counts_by_class": counter.counts,
            "totals_by_direction": counter.totals_by_direction(),
            "total_count": counter.total(),
        },
        "recall_sanity_check": {
            "note": "Tracks that came within near_miss_threshold_px of the line but were never "
            "confirmed crossing it. Spot-check these - may be real misses (track lost at the "
            "boundary) or vehicles that legitimately turned off/stopped/were still near the line "
            "at video end, not necessarily errors.",
            "threshold_px": args.near_miss_threshold_px,
            "flagged": [vars(n) for n in near_misses],
        },
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Frames processed: {frame_idx}")
    print(f"Avg FPS: {report['performance']['avg_fps']}  (median: {report['performance']['median_fps']}, "
          f"p10: {report['performance']['p10_fps']})")
    print(f"Avg latency: {report['performance']['avg_latency_ms']} ms/frame")
    print(f"\nCounts: {counter.counts}")
    print(f"Totals by direction: {counter.totals_by_direction()}")
    print(f"\nRecall sanity check ({len(near_misses)} flagged near-misses, threshold={args.near_miss_threshold_px}px):")
    for n in near_misses:
        print(f"  track {n.track_id} ({n.class_name}): min dist {n.min_abs_distance_px}px, seen {n.frames_seen} frames")
    print(f"\nSaved report to: {output_path}")


if __name__ == "__main__":
    main()
