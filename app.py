"""Streamlit frontend: upload a traffic video, run detection + tracking +
line-crossing counting on it, and show/download the annotated result.

Designed to work both locally (GPU, full video) and on Streamlit
Community Cloud (CPU-only, free tier) - resolve_device() already falls
back to CPU automatically, and model weights (not committed to git) are
fetched from the GitHub Release on first run if missing.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import altair as alt
import cv2
import pandas as pd
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
    "car": "#16a34a",
    "truck": "#ea580c",
    "bus": "#dc2626",
    "motorcycle": "#2563eb",
    "bicycle": "#9333ea",
}

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

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
.block-container { padding-top: 1.5rem; max-width: 1180px; }

/* ---- Hero header ---- */
.hero {
    background: linear-gradient(120deg, #0f172a 0%, #164e63 55%, #0ea5a4 100%);
    border-radius: 18px;
    padding: 2.4rem 2.4rem;
    margin-bottom: 1.5rem;
    color: #f8fafc;
    box-shadow: 0 12px 32px -12px rgba(14, 116, 144, 0.45);
    animation: fadeInDown 0.5s ease-out;
}
.hero h1 { margin: 0; font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; }
.hero p { margin: 0.6rem 0 0 0; color: #cbd5e1; font-size: 1rem; max-width: 640px; line-height: 1.5; }
.hero-badges { margin-top: 1.1rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }
.hero-badge {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 999px;
    padding: 0.3rem 0.9rem;
    font-size: 0.78rem;
    font-weight: 600;
    color: #e2e8f0;
    backdrop-filter: blur(4px);
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] { background: #0f172a; }
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,0.04);
    border-color: rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
}
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12); }

/* ---- Buttons ---- */
.stButton > button {
    border-radius: 10px;
    font-weight: 700;
    padding: 0.65rem 1.5rem;
    border: none;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0ea5a4, #2563eb);
    box-shadow: 0 4px 14px -4px rgba(37, 99, 235, 0.5);
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 10px 22px -6px rgba(37, 99, 235, 0.55); }
.stButton > button:active { transform: translateY(0); }

/* ---- File uploader ---- */
[data-testid="stFileUploaderDropzone"] {
    border-radius: 14px;
    border: 2px dashed #94a3b8;
    transition: border-color 0.2s ease, background 0.2s ease;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: #0ea5a4; background: rgba(14,165,164,0.05); }

/* ---- Bordered containers (cards) ---- */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px !important;
    transition: box-shadow 0.2s ease;
}

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

/* ---- Stat cards ---- */
.stat-card { text-align: center; padding: 0.3rem 0 0.1rem 0; animation: fadeInUp 0.45s ease-out both; }
.stat-card .icon { font-size: 1.7rem; }
.stat-card .value { font-size: 1.85rem; font-weight: 800; margin-top: 0.15rem; line-height: 1.1; }
.stat-card .label {
    font-size: 0.76rem; color: #64748b; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.04em; margin-top: 0.25rem;
}

/* ---- Footer ---- */
.app-footer {
    margin-top: 2.5rem; padding-top: 1.25rem; border-top: 1px solid #e2e8f0;
    color: #64748b; font-size: 0.85rem; text-align: center;
}
.app-footer a { color: #0ea5a4; text-decoration: none; font-weight: 600; }

@keyframes fadeInDown { from { opacity: 0; transform: translateY(-12px); } to { opacity: 1; transform: translateY(0); } }
@keyframes fadeInUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
</style>
"""


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
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    tmp_path = weights_path.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    tmp_path.rename(weights_path)
    return weights_path


@st.cache_resource(show_spinner=False, max_entries=1)
def load_tracker(model_label: str, confidence: float, iou: float) -> VehicleTracker:
    # max_entries=1: on a memory-constrained deployment (Streamlit Community Cloud's
    # free tier is ~1GB RAM), switching model/confidence/iou between runs would
    # otherwise leave every previous PyTorch model resident in memory forever -
    # st.cache_resource keeps entries indefinitely by default. Capping at 1 evicts
    # the old one before loading the new one.
    base_config = load_config()
    model_info = MODEL_OPTIONS[model_label]
    weights_path = ensure_weights(model_info["filename"])

    config = {
        **base_config,
        "model": {**base_config["model"], "weights": str(weights_path), "confidence": confidence, "iou": iou},
        "classes": {"vehicle_class_ids": model_info["vehicle_class_ids"], "names": model_info["names"]},
    }
    return VehicleTracker(config), config


def transcode_for_browser(mp4v_path: str) -> str:
    """OpenCV's VideoWriter (via opencv-python's PyPI wheel) has no H.264
    encoder available - libx264 is GPL-licensed and excluded from the
    prebuilt wheel - so it can only write MPEG-4 Part 2 ("mp4v"), which
    plays fine in desktop players like VLC but which browsers' native
    <video> element (what st.video renders) generally won't play inline.
    Re-encoding through the system ffmpeg binary to H.264/yuv420p with
    +faststart fixes browser playback. Falls back to the original file
    (with a warning) if ffmpeg isn't available, rather than crashing."""
    if shutil.which("ffmpeg") is None:
        st.warning("ffmpeg not found - video was written in a format most browsers won't play inline. "
                    "Download it to view, or install ffmpeg and rerun.")
        return mp4v_path

    h264_path = str(Path(mp4v_path).with_suffix("")) + "_h264.mp4"
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", mp4v_path, "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", "-loglevel", "error", h264_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not Path(h264_path).exists():
        st.warning("Video re-encoding for browser playback failed - download it to view instead.")
        return mp4v_path
    return h264_path


MAX_PROCESSING_DIMENSION = 1280  # cap the longer side of the frame before running anything


def _scaled_dims(width: int, height: int, max_dim: int = MAX_PROCESSING_DIMENSION) -> tuple[int, int]:
    scale = min(1.0, max_dim / max(width, height))
    # round to even numbers - required for yuv420p (used by the H.264 transcode step)
    new_w = max(2, int(width * scale) // 2 * 2)
    new_h = max(2, int(height * scale) // 2 * 2)
    return new_w, new_h


def process_video(video_path: str, config: dict, line_y: float, max_frames: int, progress_bar) -> tuple[str, dict]:
    tracker, _ = load_tracker(st.session_state["model_label"], config["model"]["confidence"], config["model"]["iou"])

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width, height = _scaled_dims(orig_width, orig_height)
    needs_resize = (width, height) != (orig_width, orig_height)
    if needs_resize:
        st.write(f"Downscaling {orig_width}×{orig_height} → {width}×{height} for processing "
                 f"(keeps memory/CPU use reasonable, especially on CPU-only deployments)...")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = min(total_frames, max_frames) if max_frames else total_frames

    line_config = {**config, "counting": {
        **config["counting"],
        "line": {**config["counting"]["line"], "point1": [0.0, line_y], "point2": [1.0, line_y]},
    }}
    counter = LineCounter(line_config, width, height)

    raw_out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    writer = cv2.VideoWriter(raw_out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_idx = 0
    inference_times = []
    unique_track_ids = set()
    peak_in_frame = 0
    while frame_idx < frames_to_process:
        ok, frame = cap.read()
        if not ok:
            break
        if needs_resize:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

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
            progress_bar.progress(frame_idx / frames_to_process, text=f"Analyzing frame {frame_idx}/{frames_to_process}")

    cap.release()
    writer.release()

    playable_path = transcode_for_browser(raw_out_path)

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
    return playable_path, results


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


def stat_card(col, icon: str, label: str, value, accent: str = "#0f172a"):
    with col:
        with st.container(border=True):
            st.markdown(
                f"""<div class="stat-card">
                    <div class="icon">{icon}</div>
                    <div class="value" style="color:{accent};">{value}</div>
                    <div class="label">{label}</div>
                </div>""",
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


def render_class_chart(counts_by_class: dict):
    rows = [{"class": cls_name.capitalize(), "count": sum(d.values())} for cls_name, d in counts_by_class.items()]
    df = pd.DataFrame(rows)
    color_scale = alt.Scale(
        domain=[r["class"] for r in rows],
        range=[CLASS_COLORS.get(r["class"].lower(), "#64748b") for r in rows],
    )
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
        .encode(
            x=alt.X("count:Q", title="Vehicles"),
            y=alt.Y("class:N", sort="-x", title=None),
            color=alt.Color("class:N", scale=color_scale, legend=None),
            tooltip=[alt.Tooltip("class:N", title="Class"), alt.Tooltip("count:Q", title="Count")],
        )
        .properties(height=alt.Step(40))
    )
    st.altair_chart(chart, use_container_width=True)


def render_how_it_works():
    st.write("")
    cols = st.columns(3)
    steps = [
        ("📹", "1. Upload footage", "Drop in a traffic video — dashcam, CCTV, or drone footage all work."),
        ("🧠", "2. AI does the work", "YOLO26n detects vehicles, ByteTrack follows each one across frames."),
        ("📊", "3. Get real counts", "Every vehicle is counted once, the moment it crosses your line — with direction."),
    ]
    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            with st.container(border=True):
                st.markdown(
                    f"""<div style="text-align:center; padding: 0.5rem 0.25rem;">
                        <div style="font-size:2.1rem;">{icon}</div>
                        <div style="font-weight:700; margin-top:0.5rem; font-size:0.98rem;">{title}</div>
                        <div style="color:#64748b; font-size:0.86rem; margin-top:0.35rem; line-height:1.4;">{desc}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )


def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()))
        st.session_state["model_label"] = model_label

        with st.container(border=True):
            st.markdown("**🎯 Detection**")
            confidence = st.slider("Confidence threshold", 0.05, 0.9, 0.35, 0.05)
            iou = st.slider("IoU threshold (NMS)", 0.1, 0.9, 0.5, 0.05)

        with st.container(border=True):
            st.markdown("**📏 Counting line**")
            line_y = st.slider("Position (% down the frame)", 0, 100, 82) / 100.0
            st.caption("Runs horizontally at this height. Place it where vehicles are large and clearly visible.")

        with st.container(border=True):
            st.markdown("**⏱️ Performance**")
            max_frames = st.slider("Max frames to process", 30, 1000, 300, 30)
            st.caption("Caps runtime — especially important on CPU-only deployments.")

        st.divider()
        st.caption("Built with YOLO26 + ByteTrack")
        st.caption("[View source on GitHub](https://github.com/gayathripalakurthi/vehicle-detection-counting)")

    return model_label, confidence, iou, line_y, max_frames


def render_results(results: dict, video_path: str):
    st.success(f"✅ Done — processed **{results['frames_processed']} frames** at "
               f"**{results['avg_processing_fps']} FPS**")

    tab_video, tab_stats, tab_raw = st.tabs(["🎥 Annotated Video", "📊 Statistics", "📄 Raw JSON"])

    with tab_video:
        vcol, _ = st.columns([3, 1])
        with vcol:
            st.video(video_path)
        with open(video_path, "rb") as f:
            st.download_button("⬇ Download annotated video", f, file_name="processed.mp4",
                                mime="video/mp4", use_container_width=True)

    with tab_stats:
        totals = results["totals_by_direction"]
        c1, c2, c3, c4, c5 = st.columns(5)
        stat_card(c1, "🎯", "Total Crossed", results["total_count"], "#0ea5a4")
        stat_card(c2, "⬅️", "IN", totals["IN"], "#2563eb")
        stat_card(c3, "➡️", "OUT", totals["OUT"], "#ea580c")
        stat_card(c4, "🚘", "Unique Tracked", results["unique_vehicles_tracked"], "#0f172a")
        stat_card(c5, "👀", "Peak In Frame", results["peak_vehicles_in_frame"], "#0f172a")

        st.write("")
        with st.container(border=True):
            st.markdown("**Per-class breakdown**")
            render_class_chips(results["counts_by_class"])
            if results["counts_by_class"]:
                render_class_chart(results["counts_by_class"])

        with st.expander("What's the difference between \"Total crossed\" and \"Unique tracked\"?"):
            st.caption(
                "**\"Total crossed\"** counts a vehicle once, the moment it passes the line — "
                "not once per frame it's visible in. **\"Unique vehicles tracked\"** is every "
                "distinct vehicle the model ever saw, whether or not it reached the line. If that "
                "number is much bigger than the crossing total, most vehicles simply never crossed "
                "the line's position in this clip — try moving the line slider or processing more frames."
            )

    with tab_raw:
        st.json(results)
        st.download_button("⬇ Download results JSON", json.dumps(results, indent=2),
                            file_name="counts.json", mime="application/json")


def main():
    st.set_page_config(page_title="Vehicle Detection & Counting", page_icon="🚦", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    render_hero()

    model_label, confidence, iou, line_y, max_frames = render_sidebar()

    uploaded = st.file_uploader("📹 Upload a video", type=["mp4", "avi", "mov"])

    if uploaded is None:
        st.info("👆 Upload a video to get started, or try one of the sample clips in `data/raw/` if running locally.")
        render_how_it_works()
        st.markdown(
            """<div class="app-footer">Vehicle Detection &amp; Counting System · YOLO26n fine-tuned on VisDrone2019-DET ·
            <a href="https://github.com/gayathripalakurthi/vehicle-detection-counting" target="_blank">View source on GitHub</a></div>""",
            unsafe_allow_html=True,
        )
        return

    with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as tmp_in:
        tmp_in.write(uploaded.read())
        input_path = tmp_in.name

    with st.container(border=True):
        pcol1, pcol2 = st.columns([2, 1])
        with pcol1:
            st.markdown(f"**Ready to process:** `{uploaded.name}` ({uploaded.size / 1e6:.1f} MB)")
            st.video(input_path)
        with pcol2:
            st.write("")
            st.write("")
            run_clicked = st.button("▶  Run detection + tracking + counting", type="primary", use_container_width=True)

    if run_clicked:
        with st.status("Processing video...", expanded=True) as status:
            st.write("Loading model (downloads from the release on first use, then cached)...")
            base_config = load_config()
            model_info = MODEL_OPTIONS[model_label]
            weights_path = ensure_weights(model_info["filename"])
            config = {
                **base_config,
                "model": {**base_config["model"], "weights": str(weights_path), "confidence": confidence, "iou": iou},
                "classes": {"vehicle_class_ids": model_info["vehicle_class_ids"], "names": model_info["names"]},
            }
            st.write("Model ready. Running detection + tracking + counting...")

            progress_bar = st.progress(0.0, text="Starting...")
            output_path, results = process_video(input_path, config, line_y, max_frames, progress_bar)
            progress_bar.empty()

            st.write(f"Processed {results['frames_processed']} frames — encoding video for playback...")
            status.update(label="Done!", state="complete", expanded=False)

        render_results(results, output_path)

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
