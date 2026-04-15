"""
Sample preview: visual preview of samples with labels and raw input.
Displays images with overlaid label information using OpenCV.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from config import Config
from common.paths import get_processed_samples_path

logger = logging.getLogger("cs2_nav.preview")


def preview_samples(
    n_samples: int = 5,
    indices: Optional[list[int]] = None,
    cfg: Optional[Config] = None,
) -> None:
    """
    Preview samples by displaying images with label overlays.

    Args:
        n_samples: Number of random samples to preview.
        indices: Specific sample indices to preview (overrides n_samples).
        cfg: Configuration.
    """
    cfg = cfg or Config()

    samples_path = get_processed_samples_path(cfg)
    if not samples_path.exists():
        logger.error(f"Samples file not found: {samples_path}")
        return

    df = pd.read_parquet(str(samples_path), engine="pyarrow")
    if df.empty:
        logger.error("No samples to preview")
        return

    # Select samples
    if indices is not None:
        selected = df.iloc[indices]
    else:
        n = min(n_samples, len(df))
        selected = df.sample(n=n, random_state=cfg.export.random_seed)

    for idx, (_, row) in enumerate(selected.iterrows()):
        _show_sample(row, cfg)


def _show_sample(row: pd.Series, cfg: Config) -> None:
    """Display a single sample with label overlay."""
    image_path = Path(cfg.paths.dataset_dir) / row["image_path"]

    if not image_path.exists():
        logger.warning(f"Image not found: {image_path}")
        return

    frame = cv2.imread(str(image_path))
    if frame is None:
        logger.warning(f"Could not read image: {image_path}")
        return

    # Build overlay text
    lines = [
        f"ID: {row['sample_id']}",
        f"Frame: {row['frame_index']}",
        f"Map: {row.get('map_name', 'unknown')}",
        f"Scenario: {row.get('scenario_type', 'navigation')}",
        "",
        f"MOVE: {row.get('action_move', 'N/A')}",
        f"TURN: {row.get('action_turn', 'N/A')}",
        f"JUMP: {row.get('action_jump', 0)}",
        f"CROUCH: {row.get('action_crouch', 0)}",
        f"FIRE: {row.get('action_fire', 0)}",
        f"MACRO: {row.get('action_macro', 'N/A')}",
        "",
        f"Keys: W={row.get('key_w',0)} A={row.get('key_a',0)} "
        f"S={row.get('key_s',0)} D={row.get('key_d',0)}",
        f"Mouse: dx={row.get('mouse_dx',0):.1f} dy={row.get('mouse_dy',0):.1f}",
        f"Duration: {row.get('duration_ms',0):.1f}ms",
    ]

    # Draw text on image
    y_offset = 20
    for line in lines:
        cv2.putText(
            frame, line, (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )
        y_offset += 18

    # Show image
    cv2.imshow(f"Sample: {row['sample_id']}", frame)
    logger.info(f"Showing sample {row['sample_id']} — press any key to continue")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
