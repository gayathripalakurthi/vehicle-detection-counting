# Vehicle Detection and Counting System

A vehicle detection, tracking, and line-crossing counting pipeline for traffic video. Detects vehicles frame-by-frame, assigns persistent track IDs across frames, and counts each vehicle exactly once as it crosses a configurable line — with per-class and directional (IN/OUT) breakdowns.

**Status: Stage A (pretrained baseline) complete.** Detection, tracking, counting, and evaluation all work end-to-end on sample footage. Stage B (fine-tuning on a labeled dataset) has not started yet.

## Architecture

```
Video -> OpenCV frame read -> YOLO26n detector (vehicle classes only)
      -> ByteTrack multi-object tracker (persistent IDs)
      -> Line-crossing logic (once per track ID, hysteresis-guarded)
      -> Counting -> Visualization overlay -> Output video + JSON results
```

**Why these choices:**
- **YOLO26n** (Ultralytics, Jan 2026): current recommended default, NMS-free (no DFL post-processing), faster CPU/GPU inference than YOLO11 at comparable accuracy — a good fit for a 4GB VRAM GPU.
- **ByteTrack**: ships with Ultralytics at no extra dependency cost, simple and fast; BoT-SORT is available as a drop-in alternative (`tracking.tracker: botsort` in config) for later comparison experiments.
- **Custom line-crossing logic** rather than a library (e.g. `supervision`): written by hand in `src/counting/` since understanding the crossing/counting logic is a core goal of this project, not just gluing components together.

## Setup

Requires **Python 3.11** specifically — not whatever `python` resolves to by default. As of this writing, `opencv-python` has no wheels for Python 3.14, so a newer default Python interpreter will not work here.

```bash
python -m venv .venv
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\pip install -r requirements.txt
```

The first command installs PyTorch separately with a CUDA-specific wheel index (see the comment at the top of `requirements.txt` — installing torch via the default index pulls a CPU-only build). Swap `cu130` for `cpu` if you don't have an NVIDIA GPU.

Also needs **FFmpeg** on PATH (OpenCV's video backend) — `winget install Gyan.FFmpeg` on Windows.

Verify the GPU is visible:

```bash
.venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Developed and tested against an RTX 2050 (4GB VRAM) — all default settings (`yolo26n`, 640px input) are sized for that class of GPU.

## Usage

All scripts read defaults from `configs/config.yaml`; any can be overridden with `--source`.

```bash
# Sanity-check the detector on a handful of sampled frames
python scripts/run_detection.py --source data/raw/your_video.mp4 --frames 15

# Detection + tracking, annotated output video only
python scripts/run_tracking.py --source data/raw/your_video.mp4

# Full pipeline: detect + track + line-crossing count, video + JSON results
python scripts/run_counting.py --source data/raw/your_video.mp4

# Runtime performance stats + recall sanity check (see Evaluation below)
python scripts/evaluate.py --source data/raw/your_video.mp4
```

Outputs land in `outputs/videos/` (annotated video) and `outputs/results/` (JSON counts and evaluation reports) unless overridden with `--output`/`--results`.

## Configuration

Everything tunable lives in `configs/config.yaml` — no hardcoded settings in source:

- `model`: which YOLO weights, confidence/IoU thresholds, input resolution, device
- `classes`: which COCO class IDs count as "vehicle" (bicycle, car, motorcycle, bus, truck)
- `tracking`: which tracker (`bytetrack`/`botsort`) and its thresholds
- `counting.line`: the counting line's two endpoints as **normalized** (0–1) coordinates, so it scales with any video resolution, plus a `margin_px` dead zone (see Limitations)
- `io`: input/output paths

## Evaluation

Full detection-quality metrics (mAP, precision/recall against ground-truth boxes) need a labeled validation set, which doesn't exist yet for this project — that's part of Stage B (dataset selection, Phase 12+). In the meantime, `scripts/evaluate.py` reports:

- **Runtime performance**: average/median/p10 FPS and per-frame latency
- **Recall sanity check**: flags any track that got close to the counting line but was never confirmed crossing it — a cheap proxy for missed counts without hand-labeled ground truth. Not every flag is a real miss (a vehicle can legitimately turn off or still be near the line when the video ends), so treat it as a shortlist to spot-check, not an error count.

Current baseline numbers on the sample clip (RTX 2050, `yolo26n`, 640px, bytetrack): **~47 FPS avg** (median 47.2, p10 41.2), ~27ms/frame latency, 438 frames processed.

## Sample data

`data/raw/` is gitignored (see Model weights below for the equivalent weights situation). Two clips were used during development, both sourced from Pixabay (free-to-use, no attribution required):

- `sample_traffic_elevated.mp4` — the working baseline clip: an elevated, oblique CCTV-style angle over a multi-lane boulevard.
- `sample_traffic.mp4` — a straight-down (nadir) drone shot, kept specifically as a failure-case example (see below), not used as a baseline.

## Failure cases

Two concrete failure modes found and documented so far (see `outputs/plots/failure_cases/` for the generated evidence images — regenerate with `python scripts/generate_failure_examples.py`):

1. **Camera angle (nadir/top-down view)**: `sample_traffic.mp4` is a straight-down drone shot. A COCO-pretrained detector is trained almost entirely on street/eye-level imagery, so on this clip vehicle confidence collapses to ~0.02–0.08 — the model finds the right shape but far below any usable threshold (0.35 default). This isn't a bug; it's a real limitation of using a COCO-pretrained model on an unfamiliar viewing angle, and is exactly the kind of gap Stage B fine-tuning on angle-diverse data would need to address.

2. **Track loss near the counting line**: found via `evaluate.py`'s recall check, not by inspection. A vehicle (track 54) was tracked steadily for 80+ frames at borderline 0.35–0.57 confidence, closing in on the counting line — then the track dropped 13.5px short of it, producing a genuine missed count. This is a tracker/detector confidence limitation, not a counting-logic bug (verified by tracing the track's position frame-by-frame).

A third thing worth knowing, not a failure but a correctness note: the baseline clip's road has a median strip splitting it into two carriageways. One tracked vehicle was found moving in the opposite direction to the rest (confirmed via a 30-frame trajectory trace and visible taillight orientation) and was correctly counted as the opposite direction (`IN` vs `OUT`) — a working example of the bidirectional counting the line-crossing logic is designed for, not a bug.

## Project structure

```
vehicle-detection-counting/
├── configs/            # YAML configuration (model, tracking, counting line, training)
├── data/                # raw/processed video, dataset splits (gitignored)
├── models/weights/      # model weight files (gitignored, auto-downloaded on first run)
├── src/
│   ├── detection/       # YOLO detector wrapper, vehicle-class filtering
│   ├── tracking/        # ByteTrack/BoT-SORT wrapper, persistent track IDs
│   ├── counting/        # line-crossing counting logic
│   ├── visualization/   # box/HUD/line drawing
│   ├── evaluation/      # runtime performance + recall sanity checks
│   └── utils/           # config loading, path resolution
├── scripts/             # CLI entry points (run_detection, run_tracking, run_counting, evaluate, generate_failure_examples)
├── training/            # fine-tuning scripts (Stage B, not started)
├── tests/                # unit tests (not started)
├── outputs/              # processed videos, JSON results, plots (gitignored)
└── notebooks/            # exploratory analysis (unused so far)
```

## Model weights

Weight files are not committed to git (see `.gitignore`). Pretrained YOLO26 weights are downloaded automatically by Ultralytics on first run. Fine-tuned weights (Stage B) will be published as a GitHub Release rather than committed directly.

## Limitations

- No custom fine-tuning yet — running purely on COCO-pretrained weights, so anything far from COCO's training distribution (nadir angles, night/rain scenes, unusual vehicle types) is unvalidated at best (see Failure cases above for a concrete example).
- No labeled validation set yet, so there are no mAP/precision/recall numbers — only runtime performance and a heuristic recall check.
- Only tested against two short clips from one city's traffic camera style; generalization to other camera setups is unverified.
- Counting accuracy has been spot-checked by hand (frame-by-frame trajectory tracing), not validated against an independently-annotated ground-truth count.

## Future work

Roughly in priority order (see project plan for the full list): dataset selection and fine-tuning (Stage B), broader failure-case coverage (occlusion, night/rain, dense traffic), multiple counting lines / region-based counting, vehicle speed and direction estimation, a Streamlit dashboard, real-time webcam input.
