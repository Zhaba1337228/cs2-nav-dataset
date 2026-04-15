#!/usr/bin/env python3
"""
Integration test: creates synthetic session data and runs the full
build-samples -> label-samples -> export-dataset pipeline.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from common.utils import now_iso, save_json
from common.paths import get_frames_dir, get_events_path, get_session_meta_path


def create_synthetic_session(session_id: str, n_frames: int = 50, cfg: Config | None = None):
    """Create a fake session with frames and events for testing."""
    cfg = cfg or Config()
    cfg.paths.ensure_dirs()

    session_dir = get_frames_dir(session_id, cfg).parent
    frames_dir = get_frames_dir(session_id, cfg)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy frames (small black images)
    for i in range(n_frames):
        frame_path = frames_dir / f"{i:06d}.jpg"
        img = np.zeros((cfg.capture.target_height, cfg.capture.target_width, 3), dtype=np.uint8)
        import cv2
        cv2.imwrite(str(frame_path), img)

    # Create events.csv
    events_path = get_events_path(session_id, cfg)
    import time
    t_start = time.monotonic()
    interval = 1.0 / cfg.capture.fps

    with open(events_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "event_type", "key", "button", "dx", "dy"])

        t = t_start
        for i in range(n_frames):
            # Simulate some key presses
            if i % 10 < 7:  # W pressed most of the time
                writer.writerow([t, "key_down", "w", "", 0, 0])
                writer.writerow([t + interval * 0.8, "key_up", "w", "", 0, 0])

            if i % 5 == 0:  # A pressed occasionally
                writer.writerow([t, "key_down", "a", "", 0, 0])
                writer.writerow([t + interval * 0.5, "key_up", "a", "", 0, 0])

            if i % 8 == 0:  # Space pressed occasionally
                writer.writerow([t, "key_down", "space", "", 0, 0])
                writer.writerow([t + interval * 0.3, "key_up", "space", "", 0, 0])

            # Simulate mouse movement
            dx = np.random.randn() * 10
            dy = np.random.randn() * 2
            writer.writerow([t + interval * 0.1, "mouse_move", "", "", dx, dy])

            t += interval

    # Create session.json
    meta_path = get_session_meta_path(session_id, cfg)
    save_json({
        "session_id": session_id,
        "map_name": "de_dust2",
        "scenario_type": "navigation",
        "start_time": now_iso(),
        "end_time": now_iso(),
        "fps": cfg.capture.fps,
        "frame_count": n_frames,
        "event_count": 100,
    }, meta_path)

    print(f"Created synthetic session: {session_id} ({n_frames} frames)")


def run_pipeline_test():
    """Run the full pipeline on synthetic data."""
    print("=" * 60)
    print("Integration Test: Full Pipeline")
    print("=" * 60)

    cfg = Config()

    # Create synthetic data
    print("\n[1/4] Creating synthetic session data...")
    create_synthetic_session("session_0001", n_frames=50, cfg=cfg)
    create_synthetic_session("session_0002", n_frames=30, cfg=cfg)

    # Build samples
    print("\n[2/4] Building samples...")
    from processing.build_samples import build_all_samples
    samples_df = build_all_samples(cfg=cfg)
    assert not samples_df.empty, "No samples built!"
    print(f"  Built {len(samples_df)} samples from {samples_df['session_id'].nunique()} sessions")

    # Label samples
    print("\n[3/4] Labeling samples...")
    from processing.labeling import label_samples, save_label_map, compute_label_stats
    labeled_df = label_samples(samples_df, cfg)
    assert not labeled_df.empty, "No labels assigned!"
    save_label_map(cfg)
    compute_label_stats(labeled_df, cfg)
    print(f"  Labeled {len(labeled_df)} samples")
    print(f"  Move distribution: {labeled_df['action_move'].value_counts().to_dict()}")
    print(f"  Turn distribution: {labeled_df['action_turn'].value_counts().to_dict()}")

    # Export dataset
    print("\n[4/4] Exporting dataset...")
    from processing.export_dataset import export_dataset
    summary = export_dataset(labeled_df, cfg)
    assert summary, "Export failed!"
    print(f"  Train: {summary['train_samples']} samples")
    print(f"  Val: {summary['val_samples']} samples")

    # Verify output files
    print("\nVerifying output files...")
    expected_files = [
        cfg.paths.processed_dir / "samples.parquet",
        cfg.paths.processed_dir / "samples.csv",
        cfg.paths.processed_dir / "train.parquet",
        cfg.paths.processed_dir / "val.parquet",
        cfg.paths.processed_dir / "label_map.json",
        cfg.paths.processed_dir / "dataset_schema.json",
        cfg.paths.processed_dir / "splits.json",
        cfg.paths.manifests_dir / "train_manifest.jsonl",
        cfg.paths.manifests_dir / "val_manifest.jsonl",
        cfg.paths.debug_dir / "alignment_report.json",
        cfg.paths.debug_dir / "label_stats.json",
    ]
    for f in expected_files:
        assert f.exists(), f"Missing: {f}"
        print(f"  OK: {f}")

    # Test PyTorch dataset loading
    print("\nTesting PyTorch Dataset...")
    from training.dataset import NavigationDataset
    from training.label_maps import LabelEncoder

    train_manifest = cfg.paths.manifests_dir / "train_manifest.jsonl"
    if train_manifest.exists():
        ds = NavigationDataset(
            manifest_path=train_manifest,
            dataset_root=cfg.paths.dataset_dir,
            history_len=1,
            image_size=(64, 64),
            label_encoder=LabelEncoder(cfg.paths.processed_dir / "label_map.json"),
        )
        print(f"  Dataset size: {len(ds)}")
        if len(ds) > 0:
            images, targets, metadata = ds[0]
            print(f"  Image shape: {images.shape}")
            print(f"  Target keys: {list(targets.keys())}")
            print(f"  Metadata: {metadata}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline_test()
