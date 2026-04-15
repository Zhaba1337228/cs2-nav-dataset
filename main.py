#!/usr/bin/env python3
"""
CS2 Navigation Imitation Learning Dataset Pipeline

CLI entry point for the full data collection -> processing -> export pipeline.

Usage:
    python main.py record [--map MAP] [--scenario SCENARIO]
    python main.py build-samples [--session SESSION_ID]
    python main.py label-samples
    python main.py export-dataset
    python main.py inspect-session --session SESSION_ID
    python main.py preview-samples [--n N]
    python main.py full-pipeline [--map MAP] [--scenario SCENARIO]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from common.logging_utils import setup_logging
from common.paths import list_session_ids, get_train_manifest_path, get_val_manifest_path
from common.utils import generate_session_id, find_next_session_index


def cmd_record(args: argparse.Namespace, cfg: Config) -> None:
    """Start recording a session."""
    from collector.recorder import Recorder

    recorder = Recorder(cfg)

    # Apply CLI overrides
    if args.fps:
        cfg.capture.fps = args.fps
    if args.map:
        cfg.capture.window_title = args.map

    session_id = recorder.start_session(
        map_name=args.map or "unknown",
        scenario_type=args.scenario or "navigation",
    )

    print(f"\nRecording session: {session_id}")
    print(f"  Map: {args.map or 'unknown'}")
    print(f"  Scenario: {args.scenario or 'navigation'}")
    print(f"  FPS: {cfg.capture.fps}")
    print(f"  Hotkeys: F6=stop, F7=pause, F8=scenario, F9=new session")
    print(f"  Press F6 or Ctrl+C to stop recording.\n")

    try:
        # Update overlay periodically
        import time
        start = time.monotonic()
        while recorder.is_running() and not recorder._stop_event.is_set():
            recorder._update_overlay()
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\nInterrupted — stopping recording...")
    finally:
        recorder.stop_session()
        print(f"\nSession saved: {session_id}")


def cmd_build_samples(args: argparse.Namespace, cfg: Config) -> None:
    """Build per-frame samples from raw sessions."""
    from processing.build_samples import build_all_samples

    session_ids = None
    if args.session:
        session_ids = [args.session]

    df = build_all_samples(session_ids=session_ids, cfg=cfg)
    if not df.empty:
        print(f"\nBuilt {len(df)} samples from {df['session_id'].nunique()} sessions")
    else:
        print("\nNo samples built")


def cmd_label_samples(args: argparse.Namespace, cfg: Config) -> None:
    """Apply action labels to samples."""
    import pandas as pd
    from processing.labeling import label_samples, save_label_map, compute_label_stats
    from common.paths import get_processed_samples_path

    samples_path = get_processed_samples_path(cfg)
    if not samples_path.exists():
        print("ERROR: No samples found. Run build-samples first.")
        sys.exit(1)

    samples_df = pd.read_parquet(str(samples_path), engine="pyarrow")
    labeled_df = label_samples(samples_df, cfg)

    # Save labeled samples back
    labeled_df.to_parquet(str(samples_path), engine="pyarrow")
    labeled_df.to_csv(str(samples_path.with_suffix(".csv")), index=False)

    # Save label map and stats
    save_label_map(cfg)
    compute_label_stats(labeled_df, cfg)

    print(f"\nLabeled {len(labeled_df)} samples")
    print(f"Label distribution:")
    for col in ["action_move", "action_turn", "action_macro"]:
        if col in labeled_df.columns:
            print(f"\n  {col}:")
            for label, count in labeled_df[col].value_counts().head(10).items():
                print(f"    {label}: {count}")


def cmd_export_dataset(args: argparse.Namespace, cfg: Config) -> None:
    """Export final dataset with splits and manifests."""
    from processing.export_dataset import export_dataset

    summary = export_dataset(cfg=cfg)
    if summary:
        print(f"\nDataset export complete:")
        print(f"  Total samples: {summary['total_samples']}")
        print(f"  Train: {summary['train_samples']} samples ({summary['train_sessions']} sessions)")
        print(f"  Val:   {summary['val_samples']} samples ({summary['val_sessions']} sessions)")
    else:
        print("\nExport failed — no samples found")


def cmd_inspect_session(args: argparse.Namespace, cfg: Config) -> None:
    """Inspect a recording session."""
    from processing.inspect import inspect_session

    if not args.session:
        # List available sessions
        sessions = list_session_ids(cfg)
        if sessions:
            print("Available sessions:")
            for s in sessions:
                print(f"  {s}")
        else:
            print("No sessions found")
        return

    inspect_session(args.session, cfg)


def cmd_preview_samples(args: argparse.Namespace, cfg: Config) -> None:
    """Preview samples with visual overlay."""
    from processing.preview import preview_samples

    preview_samples(n_samples=args.n, cfg=cfg)


def cmd_full_pipeline(args: argparse.Namespace, cfg: Config) -> None:
    """Run the full pipeline: record -> build -> label -> export."""
    print("=" * 60)
    print("FULL PIPELINE: Record -> Build -> Label -> Export")
    print("=" * 60)

    # Step 1: Record
    print("\n[1/4] RECORDING")
    print("Start recording your gameplay session.")
    print("Press F6 or Ctrl+C when done.\n")
    cmd_record(args, cfg)

    # Step 2: Build samples
    print("\n[2/4] BUILDING SAMPLES")
    cmd_build_samples(args, cfg)

    # Step 3: Label
    print("\n[3/4] LABELING")
    cmd_label_samples(args, cfg)

    # Step 4: Export
    print("\n[4/4] EXPORTING DATASET")
    cmd_export_dataset(args, cfg)

    print("\n" + "=" * 60)
    print("Pipeline complete! Dataset is ready for training.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="CS2 Navigation Imitation Learning Dataset Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--dataset-dir", type=str, default=None, help="Override dataset directory")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # record
    p_record = subparsers.add_parser("record", help="Record a new gameplay session")
    p_record.add_argument("--map", type=str, default=None, help="Map name (e.g. de_dust2)")
    p_record.add_argument("--scenario", type=str, default="navigation", help="Scenario type")
    p_record.add_argument("--fps", type=int, default=None, help="Capture FPS")

    # build-samples
    p_build = subparsers.add_parser("build-samples", help="Build per-frame samples from raw sessions")
    p_build.add_argument("--session", type=str, default=None, help="Process specific session only")

    # label-samples
    subparsers.add_parser("label-samples", help="Apply action labels to samples")

    # export-dataset
    subparsers.add_parser("export-dataset", help="Export final dataset with splits")

    # inspect-session
    p_inspect = subparsers.add_parser("inspect-session", help="Inspect a recording session")
    p_inspect.add_argument("--session", type=str, default=None, help="Session ID to inspect")

    # preview-samples
    p_preview = subparsers.add_parser("preview-samples", help="Preview samples visually")
    p_preview.add_argument("--n", type=int, default=5, help="Number of samples to preview")

    # full-pipeline
    p_full = subparsers.add_parser("full-pipeline", help="Run full pipeline (record -> build -> label -> export)")
    p_full.add_argument("--map", type=str, default=None)
    p_full.add_argument("--scenario", type=str, default="navigation")
    p_full.add_argument("--fps", type=int, default=None)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Setup config
    cfg = Config()
    if args.dataset_dir:
        cfg.paths.dataset_dir = Path(args.dataset_dir)
        cfg.paths.ensure_dirs()

    # Setup logging
    setup_logging(level=getattr(logging, args.log_level))

    # Dispatch command
    commands = {
        "record": cmd_record,
        "build-samples": cmd_build_samples,
        "label-samples": cmd_label_samples,
        "export-dataset": cmd_export_dataset,
        "inspect-session": cmd_inspect_session,
        "preview-samples": cmd_preview_samples,
        "full-pipeline": cmd_full_pipeline,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args, cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
