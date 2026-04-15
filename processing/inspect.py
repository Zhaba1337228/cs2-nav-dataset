"""
Session inspection: debug summary for a recorded session.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from config import Config
from common.paths import (
    get_session_meta_path,
    get_events_path,
    get_frames_dir,
    get_alignment_report_path,
    get_dropped_frames_report_path,
)
from common.utils import load_json

logger = logging.getLogger("cs2_nav.inspect")
console = Console()


def inspect_session(session_id: str, cfg: Optional[Config] = None) -> dict:
    """
    Print a debug summary for a session.
    Returns a dict with session info.
    """
    cfg = cfg or Config()

    info = {"session_id": session_id, "status": "unknown"}

    # Session metadata
    meta_path = get_session_meta_path(session_id, cfg)
    if meta_path.exists():
        meta = load_json(meta_path)
        info["meta"] = meta
        info["status"] = "complete"
    else:
        info["status"] = "no_metadata"

    # Events
    events_path = get_events_path(session_id, cfg)
    if events_path.exists():
        # Count lines (minus header)
        with open(events_path, "r") as f:
            event_count = sum(1 for _ in f) - 1
        info["event_count"] = max(0, event_count)
    else:
        info["event_count"] = 0
        info["status"] = "no_events"

    # Frames
    frames_dir = get_frames_dir(session_id, cfg)
    if frames_dir.exists():
        frame_files = sorted(frames_dir.glob("*.jpg"))
        info["frame_count"] = len(frame_files)
        if frame_files:
            info["first_frame"] = frame_files[0].name
            info["last_frame"] = frame_files[-1].name
            info["frame_size_bytes"] = frame_files[0].stat().st_size
    else:
        info["frame_count"] = 0
        info["status"] = "no_frames"

    # Alignment report
    align_path = get_alignment_report_path(cfg)
    if align_path.exists():
        reports = load_json(align_path)
        if session_id in reports:
            info["alignment"] = reports[session_id]

    # Dropped frames report
    dropped_path = get_dropped_frames_report_path(cfg)
    if dropped_path.exists():
        dropped = load_json(dropped_path)
        if session_id in dropped:
            info["dropped"] = dropped[session_id]

    # Print summary
    _print_summary(info)
    return info


def _print_summary(info: dict) -> None:
    """Print a formatted summary table."""
    table = Table(title=f"Session: {info['session_id']}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Status", info.get("status", "unknown"))
    table.add_row("Frame Count", str(info.get("frame_count", 0)))
    table.add_row("Event Count", str(info.get("event_count", 0)))

    meta = info.get("meta", {})
    if meta:
        table.add_row("Map", meta.get("map_name", "unknown"))
        table.add_row("Scenario", meta.get("scenario_type", "navigation"))
        table.add_row("Start", meta.get("start_time", ""))
        table.add_row("End", meta.get("end_time", ""))
        table.add_row("Target FPS", str(meta.get("fps", 0)))

    alignment = info.get("alignment", {})
    if alignment:
        table.add_row("Frame Interval (ms)", f"{alignment.get('frame_interval_ms', 0):.1f}")
        table.add_row("Duration (s)", f"{alignment.get('duration_s', 0):.1f}")
        table.add_row("Dropped Frames", str(alignment.get("dropped_frames", 0)))

    dropped = info.get("dropped", {})
    if dropped:
        table.add_row("Drop Rate", f"{dropped.get('drop_rate', 0):.4f}")

    console.print(table)
