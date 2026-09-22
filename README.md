# Vehicle Detection and Counting System

A vehicle detection, tracking, and line-crossing counting pipeline built with YOLO26 and ByteTrack. Detects vehicles in traffic video, assigns persistent track IDs, and counts them as they cross a configurable line — with per-class and directional (IN/OUT) breakdowns.

> Status: environment setup in progress. This README fills in as each phase lands.

## Architecture

```
Video -> Frame extraction -> YOLO26 detector -> Vehicle class filter
      -> ByteTrack multi-object tracker -> Line-crossing logic
      -> Counting -> Visualization overlay -> Output video + JSON stats
```

## Setup

See [SETUP.md](SETUP.md) (added once the environment is verified) for exact install steps and hardware requirements.

## Usage

```bash
python scripts/run_counting.py --source data/raw/traffic.mp4
```

(Full CLI reference added once `run_counting.py` exists.)

## Project structure

```
vehicle-detection-counting/
├── configs/           # YAML configuration (model, tracking, counting line, training)
├── data/               # raw/processed video, dataset splits (gitignored)
├── models/weights/     # model weight files (gitignored, see Model weights below)
├── src/
│   ├── detection/      # detector wrapper
│   ├── tracking/       # tracker wrapper
│   ├── counting/       # line-crossing counting logic
│   ├── visualization/  # overlay drawing
│   ├── evaluation/     # detection + counting metrics
│   └── utils/          # config loading, shared helpers
├── scripts/            # CLI entry points
├── training/           # fine-tuning scripts
├── tests/               # unit tests
├── outputs/             # processed videos, results, plots (gitignored)
└── notebooks/           # exploratory analysis
```

## Model weights

Weight files are not committed to git (see `.gitignore`). Pretrained YOLO26 weights are downloaded automatically by Ultralytics on first run. Fine-tuned weights (Stage B) will be published as a GitHub Release rather than committed directly.

## Limitations

To be documented after failure-case analysis (Phase 16).

## Future work

To be documented after the core system is complete (see project plan, Section 15).
