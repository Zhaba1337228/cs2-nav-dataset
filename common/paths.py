"""
Path management utilities for the dataset pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from config import Config


def get_session_dir(session_id: str, cfg: Optional[Config] = None) -> Path:
    """Return the directory path for a raw session."""
    if cfg is None:
        cfg = Config()
    return cfg.paths.raw_sessions_dir / session_id


def get_frames_dir(session_id: str, cfg: Optional[Config] = None) -> Path:
    """Return the frames directory for a session."""
    return get_session_dir(session_id, cfg) / "frames"


def get_events_path(session_id: str, cfg: Optional[Config] = None) -> Path:
    """Return the events.csv path for a session."""
    return get_session_dir(session_id, cfg) / "events.csv"


def get_session_meta_path(session_id: str, cfg: Optional[Config] = None) -> Path:
    """Return the session.json metadata path."""
    return get_session_dir(session_id, cfg) / "session.json"


def get_frame_path(session_id: str, frame_index: int, cfg: Optional[Config] = None) -> Path:
    """Return the path for a specific frame image."""
    fmt = cfg.recorder.frame_filename_fmt if cfg else "{:06d}.jpg"
    return get_frames_dir(session_id, cfg) / fmt.format(frame_index)


def get_processed_samples_path(cfg: Optional[Config] = None) -> Path:
    """Return the processed samples parquet path."""
    if cfg is None:
        cfg = Config()
    return cfg.paths.processed_dir / "samples.parquet"


def get_train_parquet_path(cfg: Optional[Config] = None) -> Path:
    if cfg is None:
        cfg = Config()
    return cfg.paths.processed_dir / "train.parquet"


def get_val_parquet_path(cfg: Optional[Config] = None) -> Path:
    if cfg is None:
        cfg = Config()
    return cfg.paths.processed_dir / "val.parquet"


def get_train_manifest_path(cfg: Optional[Config] = None) -> Path:
    if cfg is None:
        cfg = Config()
    return cfg.paths.manifests_dir / "train_manifest.jsonl"


def get_val_manifest_path(cfg: Optional[Config] = None) -> Path:
    if cfg is None:
        cfg = Config()
    return cfg.paths.manifests_dir / "val_manifest.jsonl"


def get_label_map_path(cfg: Optional[Config] = None) -> Path:
    if cfg is None:
        cfg = Config()
    return cfg.paths.processed_dir / "label_map.json"


def get_dataset_schema_path(cfg: Optional[Config] = None) -> Path:
    if cfg is None:
        cfg = Config()
    return cfg.paths.processed_dir / "dataset_schema.json"


def get_splits_path(cfg: Optional[Config] = None) -> Path:
    if cfg is None:
        cfg = Config()
    return cfg.paths.processed_dir / "splits.json"


def get_alignment_report_path(cfg: Optional[Config] = None) -> Path:
    if cfg is None:
        cfg = Config()
    return cfg.paths.debug_dir / "alignment_report.json"


def get_dropped_frames_report_path(cfg: Optional[Config] = None) -> Path:
    if cfg is None:
        cfg = Config()
    return cfg.paths.debug_dir / "dropped_frames_report.json"


def get_label_stats_path(cfg: Optional[Config] = None) -> Path:
    if cfg is None:
        cfg = Config()
    return cfg.paths.debug_dir / "label_stats.json"


def list_session_ids(cfg: Optional[Config] = None) -> list[str]:
    """List all recorded session IDs from the raw_sessions directory."""
    if cfg is None:
        cfg = Config()
    if not cfg.paths.raw_sessions_dir.exists():
        return []
    return sorted(
        d.name for d in cfg.paths.raw_sessions_dir.iterdir()
        if d.is_dir() and (d / "events.csv").exists()
    )
