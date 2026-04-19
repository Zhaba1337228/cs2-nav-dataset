# CS2 Navigation Imitation Learning Dataset Pipeline

End-to-end pipeline for collecting screen capture + input data from CS2 gameplay and producing a ready-to-train imitation learning dataset for navigation bot training.

## Overview

This project captures your CS2 gameplay (screen frames + keyboard/mouse input) and converts it into a structured dataset suitable for training a vision-based navigation model. The pipeline has 4 stages:

1. **RECORD** — Capture frames and input events into raw sessions
2. **BUILD SAMPLES** — Align raw events to per-frame sample records
3. **LABEL** — Convert raw input into action labels (move, turn, jump, etc.)
4. **EXPORT** — Create train/val splits, manifests, and PyTorch-ready format

## Requirements

- Python 3.11+
- Windows (for input hooks and screen capture)
- CS2 running in windowed or borderless windowed mode

## Installation

```bash
pip install -r requirements.txt
```

## Ubuntu SSH One-Click Setup (Training)

If you only need model training on a server (no gameplay capture), use the bootstrap installer:

```bash
bash setup_all.sh
```

This starts setup in background (`nohup`): creates `.venv`, installs training deps, installs PyTorch wheels, downloads dataset archives, and writes logs to `logs/setup_*.log`.

Monitor progress:

```bash
tail -f logs/setup_*.log
```

If you already downloaded archives manually, place them in `dataset/archives/`:

```text
dataset/archives/dataset.zip
dataset/archives/raw_sessions.zip
```

Then run installer without network downloads:

```bash
bash setup_all.sh --no-download-archives
```

If archive is already outside project root (example: `/root/raw_sessions.zip`), pass explicit paths:

```bash
bash setup_all.sh --no-download-archives \
  --dataset-zip-path /root/cs2-nav-dataset/dataset/archives/dataset.zip \
  --raw-sessions-zip-path /root/raw_sessions.zip
```

## Quick Start

### Record a session

```bash
python main.py record --map de_dust2 --scenario navigation
```

While recording:
- **F6** — Stop recording
- **F7** — Pause/resume
- **F8** — Cycle scenario type
- **F9** — Start new session

### Build the full dataset

After recording one or more sessions:

```bash
# Step 1: Build per-frame samples from all raw sessions
python main.py build-samples

# Step 2: Apply action labels
python main.py label-samples

# Step 3: Export final dataset with train/val split
python main.py export-dataset
```

### Or run the full pipeline at once

```bash
python main.py full-pipeline --map de_dust2 --scenario navigation
```

### Inspect and preview

```bash
# List and inspect sessions
python main.py inspect-session
python main.py inspect-session --session session_0001

# Preview samples visually
python main.py preview-samples --n 5
```

## Project Structure

```
cs2_nav_dataset/
├── main.py                    # CLI entry point
├── config.py                  # All configurable parameters
├── requirements.txt
├── README.md
├── DATA_COLLECTION_GUIDE.md   # How to collect good data
├── LABELING_GUIDE.md          # How labels are generated
├── QA_CHECKLIST.md            # Quality checklist
├── collector/
│   ├── capture.py             # Screen capture (mss)
│   ├── input_logger.py        # Keyboard + mouse hooks (pynput)
│   ├── recorder.py            # Orchestrator
│   ├── writer.py              # Async disk writer
│   └── overlay.py             # Rich status overlay
├── processing/
│   ├── build_samples.py       # Raw events → per-frame samples
│   ├── labeling.py            # Samples → action labels
│   ├── export_dataset.py      # Train/val split, manifests
│   ├── inspect.py             # Session inspection
│   └── preview.py             # Visual sample preview
├── training/
│   ├── dataset.py             # PyTorch Dataset class
│   ├── dataloader_utils.py    # DataLoader helpers
│   ├── train_stub.py          # Demo training loop
│   └── label_maps.py          # Label encoding maps
└── common/
    ├── schemas.py             # Data structures
    ├── utils.py               # Helper functions
    ├── paths.py               # Path management
    └── logging_utils.py       # Logging setup
```

## Output Dataset Format

After running the full pipeline:

```
dataset/
├── raw_sessions/
│   └── session_0001/
│       ├── frames/
│       │   ├── 000000.jpg
│       │   └── 000001.jpg
│       ├── events.csv
│       └── session.json
├── processed/
│   ├── samples.parquet
│   ├── samples.csv
│   ├── train.parquet
│   ├── val.parquet
│   ├── label_map.json
│   ├── dataset_schema.json
│   └── splits.json
├── manifests/
│   ├── train_manifest.jsonl
│   └── val_manifest.jsonl
└── debug/
    ├── alignment_report.json
    ├── dropped_frames_report.json
    └── label_stats.json
```

## Using the Dataset in PyTorch

```python
from training.dataset import NavigationDataset
from training.dataloader_utils import create_dataloaders

train_loader, val_loader, train_ds, val_ds = create_dataloaders(
    train_manifest="dataset/manifests/train_manifest.jsonl",
    val_manifest="dataset/manifests/val_manifest.jsonl",
    dataset_root="dataset/",
    history_len=3,          # use 3-frame sequences
    image_size=(224, 224),
    batch_size=32,
)

for images, targets, metadata in train_loader:
    # images: (B, history_len, C, H, W)
    # targets: dict with action_move, action_turn, etc.
    pass
```

## Configuration

Edit `config.py` to change:
- Capture FPS, resolution, JPEG quality
- Labeling thresholds (mouse sensitivity, etc.)
- Train/val split ratio
- Sequence history length

## Guides

- [DATA_COLLECTION_GUIDE.md](DATA_COLLECTION_GUIDE.md) — How to collect good training data
- [LABELING_GUIDE.md](LABELING_GUIDE.md) — How raw input becomes action labels
- [QA_CHECKLIST.md](QA_CHECKLIST.md) — Quality checklist for sessions and datasets
