"""
Labeling module: convert raw input samples into action labels for imitation learning.

Applies configurable thresholds to mouse movement and key states to produce
categorical action labels for each sample.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import pandas as pd

from config import Config, LabelingConfig
from common.schemas import (
    ACTION_MOVE_LABELS,
    ACTION_TURN_LABELS,
    ACTION_MACRO_LABELS,
)
from common.paths import get_processed_samples_path, get_label_map_path, get_label_stats_path

logger = logging.getLogger("cs2_nav.labeling")


def classify_move(row: pd.Series) -> str:
    """Classify movement direction based on key states."""
    w = row.get("key_w", 0)
    a = row.get("key_a", 0)
    s = row.get("key_s", 0)
    d = row.get("key_d", 0)

    if w and a:
        return "forward_left"
    if w and d:
        return "forward_right"
    if s and a:
        return "back_left"
    if s and d:
        return "back_right"
    if w:
        return "forward"
    if s:
        return "back"
    if a:
        return "left"
    if d:
        return "right"
    return "stop"


def classify_turn(row: pd.Series, cfg: LabelingConfig) -> str:
    """Classify turn direction and magnitude based on mouse movement."""
    dx = row.get("mouse_dx", 0.0)
    dy = row.get("mouse_dy", 0.0)
    abs_dx = row.get("mouse_abs_dx_sum", 0.0)
    abs_dy = row.get("mouse_abs_dy_sum", 0.0)

    # Check if mouse is essentially idle
    if abs_dx < cfg.idle_mouse_epsilon and abs_dy < cfg.idle_mouse_epsilon:
        return "no_turn"

    # Horizontal movement (turning)
    if abs_dx >= abs_dy:
        if dx < -cfg.mouse_large_threshold:
            return "turn_left_large"
        if dx < -cfg.mouse_medium_threshold:
            return "turn_left_medium"
        if dx < -cfg.mouse_small_threshold:
            return "turn_left_small"
        if dx > cfg.mouse_large_threshold:
            return "turn_right_large"
        if dx > cfg.mouse_medium_threshold:
            return "turn_right_medium"
        if dx > cfg.mouse_small_threshold:
            return "turn_right_small"
        return "no_turn"

    # Vertical movement (looking up/down)
    if dy < -cfg.mouse_medium_threshold:
        return "look_up"
    if dy > cfg.mouse_medium_threshold:
        return "look_down"
    return "no_turn"


def build_macro_action(
    move: str,
    turn: str,
    jump: int,
    crouch: int,
    fire: int,
) -> str:
    """
    Build a composite macro action label from individual action components.
    """
    if move == "stop" and turn == "no_turn" and not jump and not crouch and not fire:
        return "idle"

    # Check for unstuck pattern: keys pressed but minimal movement
    # This is handled at a higher level; here we just compose the label

    parts = []

    # Movement component
    if move != "stop":
        parts.append(f"move_{move}")

    # Turn component (only if significant)
    if turn not in ("no_turn",):
        parts.append(turn)

    # Action modifiers
    if jump:
        parts.append("jump")
    if crouch:
        parts.append("crouch")
    if fire:
        parts.append("fire")

    if not parts:
        return "idle"

    return "_".join(parts)


def label_samples(
    samples_df: Optional[pd.DataFrame] = None,
    cfg: Optional[Config] = None,
) -> pd.DataFrame:
    """
    Apply action labels to a samples DataFrame.

    If samples_df is None, loads from the processed samples parquet.
    """
    cfg = cfg or Config()

    if samples_df is None:
        samples_path = get_processed_samples_path(cfg)
        if not samples_path.exists():
            logger.error(f"Samples file not found: {samples_path}")
            return pd.DataFrame()
        samples_df = pd.read_parquet(str(samples_path), engine="pyarrow")

    if samples_df.empty:
        logger.warning("No samples to label")
        return samples_df

    logger.info(f"Labeling {len(samples_df)} samples...")

    # Classify movement
    samples_df["action_move"] = samples_df.apply(classify_move, axis=1)

    # Classify turning
    samples_df["action_turn"] = samples_df.apply(
        lambda row: classify_turn(row, cfg.labeling), axis=1
    )

    # Jump / crouch / fire (binary from key states)
    samples_df["action_jump"] = samples_df["key_space"].astype(int)
    samples_df["action_crouch"] = samples_df["key_ctrl"].astype(int)
    samples_df["action_fire"] = samples_df["mouse_left"].astype(int)

    # Build macro actions
    samples_df["action_macro"] = samples_df.apply(
        lambda row: build_macro_action(
            row["action_move"],
            row["action_turn"],
            row["action_jump"],
            row["action_crouch"],
            row["action_fire"],
        ),
        axis=1,
    )

    # Detect unstuck candidates: movement keys pressed but very little mouse movement
    # over a sustained period (simplified per-frame check)
    unstuck_mask = (
        (samples_df["is_moving_keys"] == True)
        & (samples_df["mouse_abs_dx_sum"] < cfg.labeling.idle_mouse_epsilon)
        & (samples_df["mouse_abs_dy_sum"] < cfg.labeling.idle_mouse_epsilon)
        & (samples_df["duration_ms"] > cfg.labeling.min_action_duration_ms)
    )
    samples_df.loc[unstuck_mask, "action_macro"] = "unstuck_candidate"

    # Detect back_off: S key pressed with large mouse turn away
    back_off_mask = (
        (samples_df["key_s"] == 1)
        & (samples_df["action_turn"].str.contains("turn_", na=False))
    )
    samples_df.loc[back_off_mask, "action_macro"] = "back_off"

    logger.info("Labeling complete")
    return samples_df


def save_label_map(cfg: Optional[Config] = None) -> dict:
    """Save the label-to-index mapping for training."""
    cfg = cfg or Config()

    label_map = {
        "action_move": {label: idx for idx, label in enumerate(ACTION_MOVE_LABELS)},
        "action_turn": {label: idx for idx, label in enumerate(ACTION_TURN_LABELS)},
        "action_jump": {"no": 0, "yes": 1},
        "action_crouch": {"no": 0, "yes": 1},
        "action_fire": {"no": 0, "yes": 1},
    }

    # Build macro label map from actual data if available
    samples_path = get_processed_samples_path(cfg)
    if samples_path.exists():
        df = pd.read_parquet(str(samples_path), engine="pyarrow")
        if "action_macro" in df.columns:
            unique_macros = sorted(df["action_macro"].unique().tolist())
            label_map["action_macro"] = {label: idx for idx, label in enumerate(unique_macros)}
    else:
        label_map["action_macro"] = {label: idx for idx, label in enumerate(ACTION_MACRO_LABELS)}

    label_map_path = get_label_map_path(cfg)
    label_map_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    logger.info(f"Label map saved to {label_map_path}")
    return label_map


def compute_label_stats(
    samples_df: pd.DataFrame,
    cfg: Optional[Config] = None,
) -> dict:
    """Compute label distribution statistics."""
    cfg = cfg or Config()

    stats = {
        "total_samples": len(samples_df),
        "sessions": samples_df["session_id"].nunique(),
        "label_distributions": {},
    }

    for col in ["action_move", "action_turn", "action_jump", "action_crouch", "action_fire", "action_macro"]:
        if col in samples_df.columns:
            dist = samples_df[col].value_counts().to_dict()
            # Convert keys to strings for JSON serialization
            dist = {str(k): int(v) for k, v in dist.items()}
            stats["label_distributions"][col] = dist

    # Save stats
    stats_path = get_label_stats_path(cfg)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Label stats saved to {stats_path}")
    return stats
