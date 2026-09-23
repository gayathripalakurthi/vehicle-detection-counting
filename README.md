# Vehicle Detection and Counting System

A vehicle detection, tracking, and line-crossing counting pipeline for traffic video. Detects vehicles frame-by-frame, assigns persistent track IDs across frames, and counts each vehicle exactly once as it crosses a configurable line — with per-class and directional (IN/OUT) breakdowns.

**Status: Stage B (fine-tuned model) complete and adopted as default.** Detection, tracking, counting, evaluation, dataset preparation, fine-tuning, and a baseline-vs-fine-tuned comparison all work end-to-end. See [Fine-tuning results](#fine-tuning-results-stage-b) below for the evidence behind that decision.

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

Developed and tested against an RTX 2050 (4GB VRAM) — all default settings (`yolo26n`, 640px input) are sized for that class of GPU. Fine-tuning peaked at 2.64GB VRAM usage, well within budget.

## Usage

All scripts read defaults from `configs/config.yaml`; any can be overridden with `--source`.

```bash
# Sanity-check the detector on a handful of sampled frames
python scripts/run_detection.py --source data/raw/your_video.mp4 --frames 15

# Detection + tracking, annotated output video only
python scripts/run_tracking.py --source data/raw/your_video.mp4

# Full pipeline: detect + track + line-crossing count, video + JSON results
python scripts/run_counting.py --source data/raw/your_video.mp4

# Runtime performance stats + recall sanity check
python scripts/evaluate.py --source data/raw/your_video.mp4

# Baseline vs fine-tuned model comparison (needs training/train.py run first)
python scripts/compare_models.py
```

Outputs land in `outputs/videos/` (annotated video) and `outputs/results/` (JSON counts and evaluation reports) unless overridden with `--output`/`--results`.

## Tests

```bash
python -m pytest tests/ -v
```

26 tests, all mocking out Ultralytics' `YOLO` class (or needing no model at all) so they run in well under a second with no GPU, weights, or video file needed — covering config loading, device resolution, detection-to-`Detection` conversion and class filtering, tracker ID persistence, the line-counter's crossing/duplicate-prevention/hysteresis logic, and the IoU-matching detection evaluator used for the model comparison.

## Configuration

Everything tunable lives in `configs/config.yaml` — no hardcoded settings in source:

- `model`: which YOLO weights, confidence/IoU thresholds, input resolution, device. Defaults to the fine-tuned model (see below); the config file has a comment explaining how to switch back to the pretrained baseline.
- `classes`: which class IDs count as "vehicle", and their names. **Not COCO ids** — the fine-tuned model has its own 5-class numbering (see `training/prepare_visdrone.py`).
- `tracking`: which tracker (`bytetrack`/`botsort`) and its thresholds
- `counting.line`: the counting line's two endpoints as **normalized** (0–1) coordinates, so it scales with any video resolution, plus a `margin_px` dead zone (see Limitations)
- `training`: dataset path, epochs, batch size, etc. for `training/train.py`
- `io`: input/output paths

## Fine-tuning results (Stage B)

**Dataset**: [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset) (8,629 aerial/drone images, official train/val/test-dev split by sequence — no leakage risk), chosen specifically because it directly targets the one concrete, evidenced weakness found in Stage A: the COCO-pretrained model's confidence collapses on nadir/aerial camera angles. Downloaded and converted via Ultralytics' built-in downloader (`training/prepare_visdrone.py`), then its 10 classes remapped to our 5 (van merged into car; tricycle classes dropped as not a clean fit) — see that script's docstring for the exact mapping and the reasoning against re-running it on already-remapped labels.

**Training**: `yolo26n`, fine-tuned from the same COCO-pretrained checkpoint the baseline uses (transfer learning, not from scratch), 640px, batch 8, lr0=0.01. Ran 47/50 epochs before early stopping (patience=15, peak at epoch 32), **116 minutes** total, peak VRAM 2.64GB. Full logs, per-epoch metrics, loss/mAP curves, confusion matrix, and periodic checkpoints (every 5 epochs) are in `training/runs/visdrone_finetune/` (gitignored — rerun `training/train.py` to reproduce).

**Comparison** (`scripts/compare_models.py`, full results in `outputs/results/model_comparison.json`):

On 548 VisDrone val images (precision/recall/F1 at conf≥0.35, IoU≥0.5, matched by class name so the baseline's COCO ids and the fine-tuned model's own ids compare fairly):

| Metric | Baseline | Fine-tuned |
|---|---:|---:|
| Precision | 0.899 | 0.848 |
| Recall | 0.162 | 0.530 |
| F1 | 0.275 | 0.652 |
| Avg FPS | 56.4 | 53.8 |
| Official mAP50 | — | 0.382 |
| Official mAP50-95 | — | 0.226 |

Per-class F1: car 0.37→0.76, bus 0.21→0.47, truck 0.09→0.32, motorcycle 0.01→0.37, bicycle 0.002→0.10 (bicycle stays weak — only 13k of the ~300k boxes in VisDrone are bicycles, a real class-imbalance limitation, not a bug).

The headline number is recall, not precision: the baseline wasn't wrong about what it detected, it just wasn't finding most vehicles in aerial-style imagery at all.

**On our own clips** (not just the benchmark — this is what actually matters for the deployed system):
- Nadir clip (the original failure case): **0 → 24 detections** across 5 sampled frames, avg confidence 0.67
- Elevated clip (main working scenario): **6 → 97 detections** on the same sampled frames, avg confidence 0.43 → 0.59 (higher confidence too, not just more noise — meaning previously-missed real vehicles, not spurious boxes)

Both scenarios improved with no meaningful precision cost, which is why the fine-tuned model is now the default rather than an optional extra.

**Honest caveats, not oversold as a full fix:**
- Spot-checking a specific nadir-clip frame (not just the 5-frame average) found a genuinely mixed result: some vehicles pass threshold now, most still don't at that exact frame. See `outputs/plots/failure_cases/camera_angle_nadir_view.jpg`.
- A new failure mode appeared: the fine-tuned model misclassifies a **boat** as a "truck" in the nadir clip — a domain-shift side effect of training on more aerial imagery (boat and truck silhouettes look more similar from directly above).
- The track-loss-near-line failure mode (a vehicle tracked steadily then lost right before the counting line) still happens with the fine-tuned model too — it's a generic near-threshold dropout issue, not something fine-tuning on a different dataset was going to fix.

## Sample data

`data/raw/` is gitignored (see Model weights below for the equivalent weights situation). Two clips were used during development, both sourced from Pixabay (free-to-use, no attribution required):

- `sample_traffic_elevated.mp4` — the working baseline clip: an elevated, oblique CCTV-style angle over a multi-lane boulevard.
- `sample_traffic.mp4` — a straight-down (nadir) drone shot, kept specifically as a failure-case example (see below).

## Failure cases

See `outputs/plots/failure_cases/` for generated evidence images — regenerate with `python scripts/generate_failure_examples.py` (uses whichever model `configs/config.yaml` currently points at).

1. **Camera angle (nadir/top-down view)**: improved substantially by Stage B fine-tuning but not eliminated — see the Fine-tuning results section above for the full before/after picture, including the new boat-misclassified-as-truck finding.

2. **Track loss near the counting line**: a vehicle at borderline detection confidence gets tracked for a while, closing in on the counting line, then the track drops a few pixels short — a genuine missed count. Found originally via `evaluate.py`'s recall check (not by inspection), and confirmed to still occur (with a different vehicle) after fine-tuning. `generate_failure_examples.py` finds this dynamically each run via the same recall-check logic, rather than a hardcoded track id, since which track (if any) ends up being the closest near-miss depends on which model and run produced it.

3. **Correctness note, not a failure**: the baseline clip's road has a median strip splitting it into two carriageways. A tracked vehicle moving opposite to the rest of the traffic (confirmed via a 30-frame trajectory trace and taillight orientation) was correctly counted in the opposite direction (`IN` vs `OUT`) — a working example of the bidirectional counting the line-crossing logic is designed for.

## Project structure

```
vehicle-detection-counting/
├── configs/            # YAML configuration (model, tracking, counting line, training)
├── data/                # raw/processed video, VisDrone dataset (gitignored)
├── models/weights/      # model weight files (gitignored, see Model weights below)
├── src/
│   ├── detection/       # YOLO detector wrapper, vehicle-class filtering
│   ├── tracking/        # ByteTrack/BoT-SORT wrapper, persistent track IDs
│   ├── counting/        # line-crossing counting logic
│   ├── visualization/   # box/HUD/line drawing
│   ├── evaluation/      # runtime performance, recall sanity checks, IoU-matching detection metrics
│   └── utils/           # config loading, path resolution
├── scripts/             # CLI entry points (run_detection, run_tracking, run_counting, evaluate, compare_models, generate_failure_examples)
├── training/            # dataset prep (prepare_visdrone.py) and fine-tuning (train.py, dataset.yaml, runs/)
├── tests/                # unit tests, 26 passing
├── outputs/              # processed videos, JSON results, plots (gitignored except curated failure-case images)
└── notebooks/            # exploratory analysis (unused so far)
```

## Model weights

Weight files are not committed to git (see `.gitignore`) — download them from the [v1.0-weights release](https://github.com/gayathripalakurthi/vehicle-detection-counting/releases/tag/v1.0-weights) instead:

```bash
mkdir -p models/weights
curl -L -o models/weights/yolo26n_visdrone_finetuned.pt \
  https://github.com/gayathripalakurthi/vehicle-detection-counting/releases/download/v1.0-weights/yolo26n_visdrone_finetuned.pt
curl -L -o models/weights/yolo26n.pt \
  https://github.com/gayathripalakurthi/vehicle-detection-counting/releases/download/v1.0-weights/yolo26n.pt
```

(The plain pretrained `yolo26n.pt` is also downloaded automatically by Ultralytics on first run if you skip grabbing it here — but the fine-tuned one, which is the default in `configs/config.yaml`, only exists via the release or by reproducing it yourself with `training/prepare_visdrone.py` then `training/train.py`, ~2 hours on a 4GB GPU.)

## Limitations

- Bicycle detection is weak even after fine-tuning (F1 ≈ 0.10) due to severe class imbalance in VisDrone (13k bicycle boxes vs. ~220k car boxes) — a targeted fix would need either oversampling bicycles or a supplementary dataset.
- The nadir/aerial camera-angle case is improved, not solved — see Fine-tuning results above.
- A new misclassification (boat → truck) appeared after fine-tuning on aerial imagery — not investigated further yet.
- Only tested against two short clips plus the VisDrone val benchmark; generalization to other camera setups, weather, and lighting is unverified.
- Counting accuracy has been spot-checked by hand (frame-by-frame trajectory tracing) and via the recall-check heuristic, not validated against an independently-annotated ground-truth count of a full video.
- No night/rain/dense-traffic/occlusion-specific testing yet.

## Future work

Roughly in priority order: broader failure-case coverage (occlusion, night/rain, dense traffic), addressing the bicycle class-imbalance gap, multiple counting lines / region-based counting, vehicle speed and direction estimation, a web frontend for uploading and processing videos, real-time webcam input.
