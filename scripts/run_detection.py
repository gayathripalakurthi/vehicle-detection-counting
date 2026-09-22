"""Phase 7 smoke test: run the pretrained detector on a handful of frames
from a video and print what it finds. No tracking or counting yet -
this just confirms the detector loads and produces sane output."""

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.detector import VehicleDetector
from src.utils.config import load_config, resolve_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None, help="video path (overrides config)")
    parser.add_argument("--frames", type=int, default=30, help="number of frames to sample")
    args = parser.parse_args()

    config = load_config()
    video_path = resolve_path(args.source or config["io"]["video_path"])

    print(f"Loading detector: {config['model']['name']} on device={config['model']['device']}")
    detector = VehicleDetector(config)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_fps, class_counts = [], {}
    frame_idx = 0
    step = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // args.frames) if args.frames else 1

    while frame_idx < args.frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx * step)
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.perf_counter()
        detections = detector.detect(frame)
        elapsed = time.perf_counter() - t0
        total_fps.append(1.0 / elapsed if elapsed > 0 else 0)

        for d in detections:
            class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1

        print(
            f"frame {frame_idx:3d} | {len(detections):2d} detections | "
            f"{elapsed * 1000:.1f} ms"
        )
        frame_idx += 1

    cap.release()

    print("\n--- Summary ---")
    print(f"Frames sampled: {frame_idx}")
    print(f"Avg inference FPS: {sum(total_fps) / len(total_fps):.1f}" if total_fps else "No frames processed")
    print("Detections by class (summed across sampled frames):")
    for name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
