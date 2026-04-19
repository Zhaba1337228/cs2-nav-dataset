"""
Build samples: align raw events to per-frame sample records.

Takes raw session data (frames + events.csv) and produces a per-frame
aligned sample table where each row represents one captured frame with
aggregated input events that occurred during that frame's interval.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from config import Config
from common.schemas import FrameSample, InputEvent
from common.paths import (
    get_events_path,
    get_frames_dir,
    get_session_meta_path,
    get_processed_samples_path,
    get_alignment_report_path,
    get_dropped_frames_report_path,
    list_session_ids,
)
from common.utils import load_json

logger = logging.getLogger("cs2_nav.build_samples")

# CSV column names for events
EVENT_COLUMNS = ["timestamp", "event_type", "key", "button", "dx", "dy"]


def canonicalize_key_name(key: str) -> str:
    """Map equivalent movement keys to canonical names used by labels."""
    k = str(key).strip().lower()
    key_aliases = {
        "up": "w",
        "down": "s",
        "left": "a",
        "right": "d",
    }
    return key_aliases.get(k, k)


def infer_initial_key_states(events_df: pd.DataFrame, tracked_keys: list[str]) -> dict[str, bool]:
    """
    Infer key states at session start.

    If the very first edge for a tracked key is `key_up`, we assume a missing
    `key_down` happened before logging started, so key was held at t_start.
    """
    initial = {k: False for k in tracked_keys}
    key_events = events_df[events_df["event_type"].isin(["key_down", "key_up"])].copy()
    if key_events.empty:
        return initial

    key_events["key"] = key_events["key"].apply(canonicalize_key_name)

    for key in tracked_keys:
        seq = key_events[key_events["key"] == key]
        if seq.empty:
            continue
        first_edge = seq.iloc[0]["event_type"]
        if first_edge == "key_up":
            initial[key] = True
    return initial


def load_events(events_path: Path) -> pd.DataFrame:
    """Load events from a session's events.csv."""
    # Read as strings first to avoid mixed-type inference warnings on noisy CSVs.
    df = pd.read_csv(
        events_path,
        names=EVENT_COLUMNS,
        header=0,
        dtype=str,
        low_memory=False,
    )
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["dx"] = pd.to_numeric(df["dx"], errors="coerce").fillna(0.0)
    df["dy"] = pd.to_numeric(df["dy"], errors="coerce").fillna(0.0)
    # Normalize optional text fields after dtype=str load.
    for col in ["event_type", "key", "button"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    if "key" in df.columns:
        df["key"] = df["key"].apply(canonicalize_key_name)
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def count_frames(frames_dir: Path, fmt: str = "{:06d}.jpg") -> int:
    """Count the number of frame files in a session's frames directory."""
    if not frames_dir.exists():
        return 0
    return len(list(frames_dir.glob("*.jpg")))


def build_samples_for_session(
    session_id: str,
    cfg: Optional[Config] = None,
) -> pd.DataFrame:
    """
    Build per-frame samples for a single session.

    For each frame, aggregates all input events that occurred between
    the previous frame timestamp and this frame's timestamp.
    """
    cfg = cfg or Config()

    events_path = get_events_path(session_id, cfg)
    frames_dir = get_frames_dir(session_id, cfg)
    meta_path = get_session_meta_path(session_id, cfg)

    if not events_path.exists():
        logger.error(f"Events file not found for session {session_id}")
        return pd.DataFrame()

    if not frames_dir.exists():
        logger.error(f"Frames directory not found for session {session_id}")
        return pd.DataFrame()

    # Load events
    events_df = load_events(events_path)
    if events_df.empty:
        logger.warning(f"No events found for session {session_id}")
        return pd.DataFrame()

    # Load session metadata
    map_name = "unknown"
    scenario_type = "navigation"
    if meta_path.exists():
        meta = load_json(meta_path)
        map_name = meta.get("map_name", "unknown")
        scenario_type = meta.get("scenario_type", "navigation")

    # Count frames
    frame_count = count_frames(frames_dir, cfg.recorder.frame_filename_fmt)
    if frame_count == 0:
        logger.error(f"No frames found for session {session_id}")
        return pd.DataFrame()

    # Calculate frame timestamps (evenly spaced based on target FPS)
    # We use the first and last event timestamps as bounds
    t_start = events_df["timestamp"].iloc[0]
    t_end = events_df["timestamp"].iloc[-1]
    frame_interval = (t_end - t_start) / max(frame_count - 1, 1)

    frame_timestamps = [t_start + i * frame_interval for i in range(frame_count)]

    # Build samples
    samples = []
    dropped = 0
    key_states = {
        "w": False, "a": False, "s": False, "d": False,
        "shift": False, "ctrl": False, "space": False,
    }
    # Recover likely "missing key_down at session start" cases.
    initial_states = infer_initial_key_states(events_df, list(key_states.keys()))
    key_states.update(initial_states)
    mouse_button_states = {"left": False, "right": False}

    event_idx = 0
    n_events = len(events_df)

    for frame_idx in range(frame_count):
        t_frame_start = frame_timestamps[frame_idx]
        t_frame_end = frame_timestamps[frame_idx + 1] if frame_idx + 1 < frame_count else t_end

        # Process events in this frame's interval
        mouse_dx_sum = 0.0
        mouse_dy_sum = 0.0
        mouse_abs_dx_sum = 0.0
        mouse_abs_dy_sum = 0.0
        # Track whether a key was active at any point during the frame interval.
        # This avoids false "stop" labels when key transitions happen near frame boundaries.
        frame_key_active = dict(key_states)

        while event_idx < n_events and events_df.iloc[event_idx]["timestamp"] < t_frame_end:
            row = events_df.iloc[event_idx]
            etype = row["event_type"]

            if etype == "key_down":
                key = str(row["key"]).lower()
                if key in key_states:
                    key_states[key] = True
                    frame_key_active[key] = True
            elif etype == "key_up":
                key = str(row["key"]).lower()
                if key in key_states:
                    # Key was active in this frame up until release.
                    frame_key_active[key] = True
                    key_states[key] = False
            elif etype == "mouse_button_down":
                btn = str(row["button"]).lower()
                if btn in mouse_button_states:
                    mouse_button_states[btn] = True
            elif etype == "mouse_button_up":
                btn = str(row["button"]).lower()
                if btn in mouse_button_states:
                    mouse_button_states[btn] = False
            elif etype == "mouse_move":
                dx = float(row["dx"])
                dy = float(row["dy"])
                mouse_dx_sum += dx
                mouse_dy_sum += dy
                mouse_abs_dx_sum += abs(dx)
                mouse_abs_dy_sum += abs(dy)

            event_idx += 1

        # Build sample record
        duration_ms = (t_frame_end - t_frame_start) * 1000.0
        is_moving = (
            frame_key_active["w"]
            or frame_key_active["a"]
            or frame_key_active["s"]
            or frame_key_active["d"]
        )
        is_turning = mouse_abs_dx_sum > 3.0 or mouse_abs_dy_sum > 3.0

        # Build raw action name
        move_parts = []
        if frame_key_active["w"]:
            move_parts.append("forward")
        if frame_key_active["s"]:
            move_parts.append("back")
        if frame_key_active["a"]:
            move_parts.append("left")
        if frame_key_active["d"]:
            move_parts.append("right")
        raw_action = "_".join(move_parts) if move_parts else "idle"

        sample = FrameSample(
            sample_id=f"{session_id}_{frame_idx:06d}",
            session_id=session_id,
            frame_index=frame_idx,
            timestamp_start=t_frame_start,
            timestamp_end=t_frame_end,
            duration_ms=duration_ms,
            image_path=f"raw_sessions/{session_id}/frames/{frame_idx:06d}.jpg",
            map_name=map_name,
            scenario_type=scenario_type,
            key_w=1 if frame_key_active["w"] else 0,
            key_a=1 if frame_key_active["a"] else 0,
            key_s=1 if frame_key_active["s"] else 0,
            key_d=1 if frame_key_active["d"] else 0,
            key_shift=1 if key_states["shift"] else 0,
            key_ctrl=1 if key_states["ctrl"] else 0,
            key_space=1 if key_states["space"] else 0,
            mouse_left=1 if mouse_button_states["left"] else 0,
            mouse_right=1 if mouse_button_states["right"] else 0,
            mouse_dx=mouse_dx_sum,
            mouse_dy=mouse_dy_sum,
            mouse_abs_dx_sum=mouse_abs_dx_sum,
            mouse_abs_dy_sum=mouse_abs_dy_sum,
            is_moving_keys=is_moving,
            is_turning_mouse=is_turning,
            raw_action_name=raw_action,
        )
        samples.append(sample.to_dict())

        # Verify frame file exists
        frame_path = frames_dir / f"{frame_idx:06d}.jpg"
        if not frame_path.exists():
            dropped += 1

    df = pd.DataFrame(samples)

    # Save alignment report
    report = {
        "session_id": session_id,
        "total_frames": frame_count,
        "total_events": len(events_df),
        "dropped_frames": dropped,
        "frame_interval_ms": frame_interval * 1000.0,
        "duration_s": t_end - t_start,
        "samples_built": len(df),
    }
    report_path = get_alignment_report_path(cfg)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # Append to report file (one JSON per session)
    existing_reports = {}
    if report_path.exists():
        try:
            with open(report_path, "r") as f:
                existing_reports = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing_reports = {}
    existing_reports[session_id] = report
    with open(report_path, "w") as f:
        json.dump(existing_reports, f, indent=2)

    # Save dropped frames report
    dropped_report = {
        "session_id": session_id,
        "dropped_frames": dropped,
        "total_frames": frame_count,
        "drop_rate": dropped / max(frame_count, 1),
    }
    dropped_path = get_dropped_frames_report_path(cfg)
    dropped_path.parent.mkdir(parents=True, exist_ok=True)
    existing_dropped = {}
    if dropped_path.exists():
        try:
            with open(dropped_path, "r") as f:
                existing_dropped = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing_dropped = {}
    existing_dropped[session_id] = dropped_report
    with open(dropped_path, "w") as f:
        json.dump(existing_dropped, f, indent=2)

    logger.info(
        f"Built {len(df)} samples for session {session_id} "
        f"(dropped frames: {dropped}/{frame_count})"
    )
    return df


def build_all_samples(
    session_ids: Optional[list[str]] = None,
    cfg: Optional[Config] = None,
) -> pd.DataFrame:
    """Build samples for all sessions and concatenate into one DataFrame."""
    cfg = cfg or Config()

    if session_ids is None:
        session_ids = list_session_ids(cfg)

    if not session_ids:
        logger.warning("No sessions found to process")
        return pd.DataFrame()

    all_samples = []
    for sid in session_ids:
        df = build_samples_for_session(sid, cfg)
        if not df.empty:
            all_samples.append(df)

    if not all_samples:
        logger.warning("No samples built from any session")
        return pd.DataFrame()

    combined = pd.concat(all_samples, ignore_index=True)

    # Save combined samples
    samples_path = get_processed_samples_path(cfg)
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(str(samples_path), engine="pyarrow")
    combined.to_csv(str(samples_path.with_suffix(".csv")), index=False)

    logger.info(f"Saved {len(combined)} total samples to {samples_path}")
    return combined
