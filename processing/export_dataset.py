"""
Export dataset: train/val split, manifests, schema, and stats.

Takes labeled samples and produces the final dataset structure ready for PyTorch training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import Config
from common.paths import (
    get_processed_samples_path,
    get_train_parquet_path,
    get_val_parquet_path,
    get_train_manifest_path,
    get_val_manifest_path,
    get_label_map_path,
    get_dataset_schema_path,
    get_splits_path,
)
from common.utils import save_json, append_jsonl

logger = logging.getLogger("cs2_nav.export_dataset")

# Columns that form the sample record for the dataset
DATASET_COLUMNS = [
    "sample_id", "session_id", "frame_index",
    "timestamp_start", "timestamp_end", "duration_ms",
    "image_path", "map_name", "scenario_type",
    "key_w", "key_a", "key_s", "key_d",
    "key_shift", "key_ctrl", "key_space",
    "mouse_left", "mouse_right",
    "mouse_dx", "mouse_dy", "mouse_abs_dx_sum", "mouse_abs_dy_sum",
    "is_moving_keys", "is_turning_mouse", "raw_action_name",
    "action_move", "action_turn",
    "action_jump", "action_crouch", "action_fire", "action_macro",
    "split",
]


def export_dataset(
    samples_df: Optional[pd.DataFrame] = None,
    cfg: Optional[Config] = None,
) -> dict:
    """
    Export labeled samples into the final dataset format.

    Returns a summary dict with split sizes and paths.
    """
    cfg = cfg or Config()
    cfg.paths.ensure_dirs()

    # Load samples if not provided
    if samples_df is None:
        samples_path = get_processed_samples_path(cfg)
        if not samples_path.exists():
            logger.error(f"Samples file not found: {samples_path}")
            return {}
        samples_df = pd.read_parquet(str(samples_path), engine="pyarrow")

    if samples_df.empty:
        logger.error("No samples to export")
        return {}

    logger.info(f"Exporting dataset with {len(samples_df)} samples...")

    # Train/val split (by session to avoid data leakage)
    session_ids = samples_df["session_id"].unique().tolist()
    rng = np.random.RandomState(cfg.export.random_seed)
    rng.shuffle(session_ids)

    n_val = max(1, int(len(session_ids) * cfg.export.val_split_ratio))
    val_sessions = set(session_ids[:n_val])
    train_sessions = set(session_ids[n_val:])

    # Assign splits
    samples_df["split"] = samples_df["session_id"].apply(
        lambda sid: "val" if sid in val_sessions else "train"
    )

    # Split dataframes
    train_df = samples_df[samples_df["split"] == "train"].reset_index(drop=True)
    val_df = samples_df[samples_df["split"] == "val"].reset_index(drop=True)

    # Save train/val parquet
    train_path = get_train_parquet_path(cfg)
    val_path = get_val_parquet_path(cfg)

    train_df.to_parquet(str(train_path), engine="pyarrow")
    val_df.to_parquet(str(val_path), engine="pyarrow")

    logger.info(f"Train: {len(train_df)} samples, Val: {len(val_df)} samples")

    # Build manifests (JSONL format — one JSON per line)
    train_manifest = get_train_manifest_path(cfg)
    val_manifest = get_val_manifest_path(cfg)

    # Clear existing manifests
    if train_manifest.exists():
        train_manifest.unlink()
    if val_manifest.exists():
        val_manifest.unlink()

    _write_manifest(train_df, train_manifest, cfg)
    _write_manifest(val_df, val_manifest, cfg)

    # Save splits info
    splits_info = {
        "train_sessions": sorted(train_sessions),
        "val_sessions": sorted(val_sessions),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "val_split_ratio": cfg.export.val_split_ratio,
        "random_seed": cfg.export.random_seed,
    }
    save_json(splits_info, get_splits_path(cfg))

    # Save dataset schema
    schema = _build_schema(samples_df)
    save_json(schema, get_dataset_schema_path(cfg))

    # Summary
    summary = {
        "total_samples": len(samples_df),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "train_sessions": len(train_sessions),
        "val_sessions": len(val_sessions),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
    }

    logger.info(f"Dataset export complete: {summary}")
    return summary


def _write_manifest(df: pd.DataFrame, manifest_path: Path, cfg: Config) -> None:
    """Write a JSONL manifest file from a DataFrame."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        record = {
            "sample_id": row["sample_id"],
            "session_id": row["session_id"],
            "frame_index": int(row["frame_index"]),
            "image_path": row["image_path"],
            "map_name": row.get("map_name", "unknown"),
            "scenario_type": row.get("scenario_type", "navigation"),
            "action_move": row.get("action_move", "stop"),
            "action_turn": row.get("action_turn", "no_turn"),
            "action_jump": int(row.get("action_jump", 0)),
            "action_crouch": int(row.get("action_crouch", 0)),
            "action_fire": int(row.get("action_fire", 0)),
            "action_macro": row.get("action_macro", "idle"),
            "split": row.get("split", "train"),
        }
        append_jsonl(manifest_path, record)


def _build_schema(df: pd.DataFrame) -> dict:
    """Build a dataset schema describing columns, types, and label vocabularies."""
    schema = {
        "version": "1.0",
        "columns": {},
        "label_vocabularies": {},
    }

    type_map = {
        "object": "string",
        "int64": "int",
        "int32": "int",
        "float64": "float",
        "float32": "float",
        "bool": "bool",
    }

    for col in df.columns:
        dtype = str(df[col].dtype)
        schema["columns"][col] = {
            "type": type_map.get(dtype, dtype),
            "nullable": bool(df[col].isna().any()),
        }

    # Extract label vocabularies
    for col in ["action_move", "action_turn", "action_macro"]:
        if col in df.columns:
            vocab = sorted(df[col].dropna().unique().tolist())
            schema["label_vocabularies"][col] = vocab

    return schema
