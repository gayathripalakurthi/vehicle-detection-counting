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

# Matches src/visualization/draw.py's box colors (converted BGR->hex) so the
# per-class chips in the UI visually tie back to the boxes drawn in the video.
CLASS_COLORS = {
    "car": "#3CC83C",
    "truck": "#F0823C",
    "bus": "#E63C3C",
    "motorcycle": "#3CA0E6",
    "bicycle": "#C83CC8",
}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

/* ---- Hero header ---- */
.hero {
    background: linear-gradient(120deg, #0f172a 0%, #1e3a5f 45%, #0ea5a4 100%);
    border-radius: 16px;
    padding: 2.25rem 2rem;
    margin-bottom: 1.75rem;
    color: #f8fafc;
    box-shadow: 0 10px 30px -10px rgba(14, 116, 144, 0.4);
    animation: fadeInDown 0.6s ease-out;
}
.hero h1 { margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -0.02em; }
.hero p { margin: 0.5rem 0 0 0; color: #cbd5e1; font-size: 1.02rem; }
.hero-badges { margin-top: 1rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }
.hero-badge {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 999px;
    padding: 0.28rem 0.85rem;
    font-size: 0.78rem;
    font-weight: 600;
    color: #e2e8f0;
    backdrop-filter: blur(4px);
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: #0f172a;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] { margin-top: 0.3rem; }

/* ---- Buttons ---- */
.stButton > button {
    border-radius: 10px;
    font-weight: 700;
    padding: 0.6rem 1.4rem;
    border: none;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0ea5a4, #2563eb);
    box-shadow: 0 4px 14px -4px rgba(37, 99, 235, 0.5);
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 20px -6px rgba(37, 99, 235, 0.55); }
.stButton > button:active { transform: translateY(0); }

/* ---- File uploader ---- */
[data-testid="stFileUploaderDropzone"] {
    border-radius: 14px;
    border: 2px dashed #94a3b8;
    transition: border-color 0.2s ease, background 0.2s ease;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: #0ea5a4; background: rgba(14,165,164,0.05); }

/* ---- Metric cards ---- */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    animation: fadeInUp 0.45s ease-out both;
}
[data-testid="stMetric"]:hover { transform: translateY(-3px); box-shadow: 0 8px 18px -6px rgba(15, 23, 42, 0.15); }
[data-testid="stMetricValue"] { font-weight: 800; }

/* ---- Class chips ---- */
.class-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.55rem 0.9rem;
    margin: 0.25rem 0.4rem 0.25rem 0;
    font-size: 0.92rem;
    font-weight: 600;
    color: #1e293b;
    animation: fadeInUp 0.4s ease-out both;
}
.class-chip .dot { width: 11px; height: 11px; border-radius: 50%; flex-shrink: 0; }
.class-chip .sub { color: #64748b; font-weight: 500; font-size: 0.82rem; }

/* ---- Result section fade-in ---- */
.result-section { animation: fadeInUp 0.5s ease-out both; }

/* ---- Footer ---- */
.app-footer {
    margin-top: 2.5rem;
    padding-top: 1.25rem;
    border-top: 1px solid #e2e8f0;
    color: #64748b;
    font-size: 0.85rem;
    text-align: center;
}
.app-footer a { color: #0ea5a4; text-decoration: none; font-weight: 600; }

@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-12px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
"""

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
        **config["counting"],
        "line": {**config["counting"]["line"], "point1": [0.0, line_y], "point2": [1.0, line_y]},
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


def render_hero():
    st.markdown(
        """
        <div class="hero">
            <h1>🚦 Vehicle Detection &amp; Counting</h1>
            <p>Upload traffic footage and get vehicle detection, multi-object tracking, and
            line-crossing counts — per class, with direction.</p>
            <div class="hero-badges">
                <span class="hero-badge">⚡ YOLO26n</span>
                <span class="hero-badge">🎯 ByteTrack</span>
                <span class="hero-badge">🛩️ Fine-tuned on VisDrone</span>
                <span class="hero-badge">📊 mAP50 0.382</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_class_chips(counts_by_class: dict):
    if not counts_by_class:
        st.info("No crossings detected in this window — try lowering the confidence threshold, "
                 "repositioning the counting line, or processing more frames.")
        return
    chips_html = ""
    for cls_name, d in sorted(counts_by_class.items(), key=lambda x: -sum(x[1].values())):
        color = CLASS_COLORS.get(cls_name, "#64748b")
        total = sum(d.values())
        chips_html += (
            f'<span class="class-chip"><span class="dot" style="background:{color}"></span>'
            f'{cls_name.capitalize()}: <b>{total}</b>'
            f'<span class="sub">&nbsp;(IN {d.get("IN", 0)} · OUT {d.get("OUT", 0)})</span></span>'
        )
    st.markdown(f'<div>{chips_html}</div>', unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Vehicle Detection & Counting", page_icon="🚦", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    render_hero()

    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()))
        st.session_state["model_label"] = model_label

        st.markdown("**Detection**")
        confidence = st.slider("Confidence threshold", 0.05, 0.9, 0.35, 0.05)
        iou = st.slider("IoU threshold (NMS)", 0.1, 0.9, 0.5, 0.05)

        st.markdown("**Counting line**")
        line_y = st.slider("Position (% down the frame)", 0, 100, 82) / 100.0
        st.caption("Runs horizontally at this height. Place it where vehicles are large and clearly visible.")

        st.markdown("**Performance**")
        max_frames = st.slider("Max frames to process", 30, 1000, 300, 30)
        st.caption("Caps runtime — especially important on CPU-only deployments.")

        st.divider()
        st.caption("Built with YOLO26 + ByteTrack · [GitHub repo](https://github.com/gayathripalakurthi/vehicle-detection-counting)")

    uploaded = st.file_uploader("📹 Upload a video", type=["mp4", "avi", "mov"])

    if uploaded is None:
        st.info("👆 Upload a video to get started, or try one of the sample clips in `data/raw/` if running locally.")
        return

    with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as tmp_in:
        tmp_in.write(uploaded.read())
        input_path = tmp_in.name

    run_clicked = st.button("▶  Run detection + tracking + counting", type="primary", use_container_width=False)

    if run_clicked:
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

        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        st.success(f"✅ Done — processed {results['frames_processed']} frames at "
                   f"**{results['avg_processing_fps']} FPS**")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.video(output_path)
            with open(output_path, "rb") as f:
                st.download_button("⬇ Download annotated video", f, file_name="processed.mp4",
                                    mime="video/mp4", use_container_width=True)

        with col2:
            totals = results["totals_by_direction"]
            st.metric("🎯 Total crossed the line", results["total_count"])
            m1, m2 = st.columns(2)
            m1.metric("⬅ IN", totals["IN"])
            m2.metric("➡ OUT", totals["OUT"])

            m3, m4 = st.columns(2)
            m3.metric("🚘 Unique vehicles tracked", results["unique_vehicles_tracked"])
            m4.metric("👀 Peak in frame", results["peak_vehicles_in_frame"])

            with st.expander("What's the difference between these numbers?"):
                st.caption(
                    "**\"Total crossed\"** counts a vehicle once, the moment it passes the line — "
                    "not once per frame it's visible in. **\"Unique vehicles tracked\"** is every "
                    "distinct vehicle the model ever saw, whether or not it reached the line. If that "
                    "number is much bigger than the crossing total, most vehicles simply never crossed "
                    "the line's position in this clip — try moving the line slider or processing more frames."
                )

            st.markdown("**Per-class breakdown**")
            render_class_chips(results["counts_by_class"])

            st.download_button(
                "⬇ Download results JSON",
                json.dumps(results, indent=2),
                file_name="counts.json",
                mime="application/json",
                use_container_width=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="app-footer">
            Vehicle Detection &amp; Counting System · YOLO26n fine-tuned on VisDrone2019-DET ·
            <a href="https://github.com/gayathripalakurthi/vehicle-detection-counting" target="_blank">View source on GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
