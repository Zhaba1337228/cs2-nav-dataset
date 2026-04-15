"""
General helper utilities.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    """Current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def monotonic_ms() -> float:
    """Monotonic clock in milliseconds."""
    return time.monotonic() * 1000.0


def monotonic_s() -> float:
    """Monotonic clock in seconds."""
    return time.monotonic()


def save_json(data: Any, path: Path, indent: int = 2) -> None:
    """Save data as JSON to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(path: Path) -> Any:
    """Load JSON from a file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: Path, record: dict) -> None:
    """Append a single JSON object to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    """Load all lines from a JSONL file."""
    records = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def generate_session_id(prefix: str = "session", index: int = 1) -> str:
    """Generate a session ID like session_0001."""
    return f"{prefix}_{index:04d}"


def find_next_session_index(raw_sessions_dir: Path, prefix: str = "session") -> int:
    """Find the next available session index by scanning existing sessions."""
    if not raw_sessions_dir.exists():
        return 1
    existing = sorted(raw_sessions_dir.iterdir())
    max_idx = 0
    for d in existing:
        if d.is_dir() and d.name.startswith(prefix + "_"):
            try:
                idx = int(d.name.split("_")[1])
                max_idx = max(max_idx, idx)
            except (ValueError, IndexError):
                continue
    return max_idx + 1
