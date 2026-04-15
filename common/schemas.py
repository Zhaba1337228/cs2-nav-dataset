"""
Data schemas — dataclasses representing all data structures in the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────
# Raw event types (recorded during capture)
# ─────────────────────────────────────────────

class EventType(str, Enum):
    KEY_DOWN = "key_down"
    KEY_UP = "key_up"
    MOUSE_BUTTON_DOWN = "mouse_button_down"
    MOUSE_BUTTON_UP = "mouse_button_up"
    MOUSE_MOVE = "mouse_move"


@dataclass
class InputEvent:
    """A single input event captured during recording."""
    timestamp: float           # monotonic clock timestamp
    event_type: str            # EventType value
    key: str = ""              # key name (for keyboard events)
    button: str = ""           # button name (for mouse button events)
    dx: float = 0.0            # mouse delta X (for mouse move events)
    dy: float = 0.0            # mouse delta Y (for mouse move events)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_csv_row(self) -> list:
        return [
            self.timestamp,
            self.event_type,
            self.key,
            self.button,
            self.dx,
            self.dy,
        ]

    @classmethod
    def from_csv_row(cls, row: list) -> InputEvent:
        return cls(
            timestamp=float(row[0]),
            event_type=row[1],
            key=row[2],
            button=row[3],
            dx=float(row[4]),
            dy=float(row[5]),
        )


# ─────────────────────────────────────────────
# Session metadata
# ─────────────────────────────────────────────

@dataclass
class SessionMeta:
    """Metadata for a recording session."""
    session_id: str
    map_name: str = "unknown"
    scenario_type: str = "navigation"
    start_time: str = ""       # ISO format timestamp
    end_time: str = ""
    fps: int = 15
    frame_count: int = 0
    event_count: int = 0
    paused: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> SessionMeta:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ─────────────────────────────────────────────
# Per-frame sample (after alignment)
# ─────────────────────────────────────────────

@dataclass
class FrameSample:
    """One row in the aligned sample table — one frame + aggregated input."""
    # Identifiers
    sample_id: str = ""
    session_id: str = ""
    frame_index: int = 0

    # Timing
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0
    duration_ms: float = 0.0

    # Image
    image_path: str = ""

    # Session context
    map_name: str = "unknown"
    scenario_type: str = "navigation"

    # Key states (binary 0/1)
    key_w: int = 0
    key_a: int = 0
    key_s: int = 0
    key_d: int = 0
    key_shift: int = 0
    key_ctrl: int = 0
    key_space: int = 0

    # Mouse button states (binary 0/1)
    mouse_left: int = 0
    mouse_right: int = 0

    # Aggregated mouse movement over frame interval
    mouse_dx: float = 0.0
    mouse_dy: float = 0.0
    mouse_abs_dx_sum: float = 0.0
    mouse_abs_dy_sum: float = 0.0

    # Derived flags
    is_moving_keys: bool = False
    is_turning_mouse: bool = False
    raw_action_name: str = ""

    # Action labels (filled by labeling stage)
    action_move: str = "stop"
    action_turn: str = "no_turn"
    action_jump: int = 0
    action_crouch: int = 0
    action_fire: int = 0
    action_macro: str = "idle"

    # Split assignment
    split: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────
# Label map entries
# ─────────────────────────────────────────────

ACTION_MOVE_LABELS = [
    "stop",
    "forward",
    "forward_left",
    "forward_right",
    "left",
    "right",
    "back",
    "back_left",
    "back_right",
]

ACTION_TURN_LABELS = [
    "no_turn",
    "turn_left_small",
    "turn_left_medium",
    "turn_left_large",
    "turn_right_small",
    "turn_right_medium",
    "turn_right_large",
    "look_up",
    "look_down",
]

ACTION_MACRO_LABELS = [
    "idle",
    "move_forward",
    "move_forward_left",
    "move_forward_right",
    "move_forward_turn_left_small",
    "move_forward_turn_left_medium",
    "move_forward_turn_left_large",
    "move_forward_turn_right_small",
    "move_forward_turn_right_medium",
    "move_forward_turn_right_large",
    "move_forward_look_up",
    "move_forward_look_down",
    "move_left",
    "move_right",
    "move_back",
    "move_back_left",
    "move_back_right",
    "back_off",
    "unstuck_candidate",
    "move_forward_jump",
    "move_forward_crouch",
    "move_forward_fire",
    "move_forward_right_turn_right",
    "move_forward_left_turn_left",
]
