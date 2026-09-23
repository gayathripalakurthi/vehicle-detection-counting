"""Streamlit frontend: upload a traffic video, run detection + tracking +
line-crossing counting on it, and show/download the annotated result.

Designed to work both locally (GPU, full video) and on Streamlit
Community Cloud (CPU-only, free tier) - resolve_device() already falls
back to CPU automatically, and model weights (not committed to git) are
fetched from the GitHub Release on first run if missing.
"""

import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.counting.line_counter import LineCounter
from src.tracking.tracker import VehicleTracker
from src.utils.config import load_config
from src.visualization.draw import draw_counting_line, draw_hud, draw_tracked_object

WEIGHTS_BASE_URL = "https://github.com/gayathripalakurthi/vehicle-detection-counting/releases/download/v1.0-weights"

FINETUNED_CLASS_NAMES = {0: "bicycle", 1: "car", 2: "motorcycle", 3: "bus", 4: "truck"}
COCO_CLASS_NAMES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

MODEL_OPTIONS = {
    "Fine-tuned on VisDrone (recommended)": {
        "filename": "yolo26n_visdrone_finetuned.pt",
        "vehicle_class_ids": list(FINETUNED_CLASS_NAMES.keys()),
        "names": FINETUNED_CLASS_NAMES,
    },
    "Pretrained baseline (COCO)": {
        "filename": "yolo26n.pt",
        "vehicle_class_ids": list(COCO_CLASS_NAMES.keys()),
        "names": COCO_CLASS_NAMES,
    },
}


@st.cache_resource(show_spinner=False)
def ensure_weights(filename: str) -> Path:
    """Downloads a weights file from the GitHub Release if it isn't
    already present locally. Cached per-filename so it only downloads
    once per app session (not once per rerun)."""
    weights_path = Path(__file__).resolve().parent / "models" / "weights" / filename
    if weights_path.exists():
        return weights_path

    import requests

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{WEIGHTS_BASE_URL}/{filename}"
    with st.spinner(f"Downloading {filename} (one-time)..."):
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        tmp_path = weights_path.with_suffix(".tmp")
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        tmp_path.rename(weights_path)
    return weights_path


@st.cache_resource(show_spinner=False)
def load_tracker(model_label: str, confidence: float, iou: float) -> VehicleTracker:
    base_config = load_config()
    model_info = MODEL_OPTIONS[model_label]
    weights_path = ensure_weights(model_info["filename"])

    config = {
        **base_config,
        "model": {**base_config["model"], "weights": str(weights_path), "confidence": confidence, "iou": iou},
        "classes": {"vehicle_class_ids": model_info["vehicle_class_ids"], "names": model_info["names"]},
    }
    return VehicleTracker(config), config


def process_video(video_path: str, config: dict, line_y: float, max_frames: int, progress_bar) -> tuple[str, dict]:
    tracker, _ = load_tracker(st.session_state["model_label"], config["model"]["confidence"], config["model"]["iou"])

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = min(total_frames, max_frames) if max_frames else total_frames

    line_config = {**config, "counting": {
        "line": {"point1": [0.0, line_y], "point2": [1.0, line_y], "margin_px": 15},
        "direction_labels": ["IN", "OUT"],
    }}
    counter = LineCounter(line_config, width, height)

    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_idx = 0
    inference_times = []
    unique_track_ids = set()
    peak_in_frame = 0
    while frame_idx < frames_to_process:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.perf_counter()
        tracked = tracker.track(frame)
        counter.update(tracked, frame_idx)
        inference_times.append(time.perf_counter() - t0)

        unique_track_ids.update(obj.track_id for obj in tracked)
        peak_in_frame = max(peak_in_frame, len(tracked))

        for obj in tracked:
            draw_tracked_object(frame, obj)
        draw_counting_line(frame, counter.p1, counter.p2)
        totals = counter.totals_by_direction()
        hud_lines = [
            f"Frame {frame_idx}/{frames_to_process}",
            f"In frame: {len(tracked)}",  # currently-tracked vehicles this frame, NOT the same as Total below
            f"IN: {totals['IN']}   OUT: {totals['OUT']}",
            f"Total crossed: {counter.total()}",
        ]
        draw_hud(frame, hud_lines)
        writer.write(frame)

        frame_idx += 1
        if frame_idx % 5 == 0 or frame_idx == frames_to_process:
            progress_bar.progress(frame_idx / frames_to_process, text=f"Processing frame {frame_idx}/{frames_to_process}")

    cap.release()
    writer.release()

    avg_fps = 1.0 / (sum(inference_times) / len(inference_times)) if inference_times else 0
    results = {
        "frames_processed": frame_idx,
        "avg_processing_fps": round(avg_fps, 2),
        "counts_by_class": counter.counts,
        "totals_by_direction": counter.totals_by_direction(),
        "total_count": counter.total(),
        # NOT the same as total_count: this is every distinct vehicle the tracker ever saw,
        # whether or not it crossed the line - usually much bigger than total_count, and
        # is what actually explains "I see way more cars than the count" (see UI caption).
        "unique_vehicles_tracked": len(unique_track_ids),
        "peak_vehicles_in_frame": peak_in_frame,
    }
    return out_path, results


def main():
    st.set_page_config(page_title="Vehicle Detection & Counting", layout="wide")
    st.title("Vehicle Detection and Counting")
    st.caption("Upload traffic video, detect and track vehicles, and count them crossing a configurable line.")

    with st.sidebar:
        st.header("Settings")
        model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()))
        st.session_state["model_label"] = model_label
        confidence = st.slider("Confidence threshold", 0.05, 0.9, 0.35, 0.05)
        iou = st.slider("IoU threshold (NMS)", 0.1, 0.9, 0.5, 0.05)
        line_y = st.slider("Counting line position (% down the frame)", 0, 100, 82) / 100.0
        max_frames = st.slider("Max frames to process (caps runtime, especially on CPU)", 30, 1000, 300, 30)
        st.caption("Line runs horizontally at the chosen height. Position it where vehicles are large and clearly visible for best results.")

    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded is None:
        st.info("Upload a video to get started, or try one of the sample clips in `data/raw/` if running locally.")
        return

    with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as tmp_in:
        tmp_in.write(uploaded.read())
        input_path = tmp_in.name

    if st.button("Run detection + tracking + counting", type="primary"):
        base_config = load_config()
        model_info = MODEL_OPTIONS[model_label]
        weights_path = ensure_weights(model_info["filename"])
        config = {
            **base_config,
            "model": {**base_config["model"], "weights": str(weights_path), "confidence": confidence, "iou": iou},
            "classes": {"vehicle_class_ids": model_info["vehicle_class_ids"], "names": model_info["names"]},
        }

        progress_bar = st.progress(0.0, text="Starting...")
        output_path, results = process_video(input_path, config, line_y, max_frames, progress_bar)
        progress_bar.empty()

        st.success(f"Done - processed {results['frames_processed']} frames at {results['avg_processing_fps']} FPS")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.video(output_path)
            with open(output_path, "rb") as f:
                st.download_button("Download annotated video", f, file_name="processed.mp4", mime="video/mp4")

        with col2:
            st.metric("Total crossed the line", results["total_count"])
            totals = results["totals_by_direction"]
            m1, m2 = st.columns(2)
            m1.metric("IN", totals["IN"])
            m2.metric("OUT", totals["OUT"])

            st.divider()
            m3, m4 = st.columns(2)
            m3.metric("Unique vehicles tracked", results["unique_vehicles_tracked"])
            m4.metric("Peak simultaneous in frame", results["peak_vehicles_in_frame"])
            st.caption(
                "**\"Total crossed\" vs. \"Unique vehicles tracked\":** a vehicle only counts toward "
                "\"Total crossed\" once, the moment it passes the line - it doesn't count again for "
                "every frame it's visible in. \"Unique vehicles tracked\" is every distinct vehicle the "
                "model saw at all, whether or not it ever reached the line - if that number is much "
                "bigger than the crossing total, most vehicles simply never crossed the line's position "
                "in this clip (try moving the line slider, or check a longer clip)."
            )

            st.subheader("Per-class counts")
            if results["counts_by_class"]:
                for cls_name, d in sorted(results["counts_by_class"].items(), key=lambda x: -sum(x[1].values())):
                    st.write(f"**{cls_name}**: {sum(d.values())} (IN: {d['IN']}, OUT: {d['OUT']})")
            else:
                st.write("No crossings detected - try adjusting the counting line position above.")

            st.download_button(
                "Download results JSON",
                json.dumps(results, indent=2),
                file_name="counts.json",
                mime="application/json",
            )


if __name__ == "__main__":
    main()
