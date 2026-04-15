"""
Label encoding maps for training.
Provides label-to-index and index-to-label mappings for all action categories.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from common.schemas import ACTION_MOVE_LABELS, ACTION_TURN_LABELS

# Binary labels
JUMP_LABELS = ["no", "yes"]
CROUCH_LABELS = ["no", "yes"]
FIRE_LABELS = ["no", "yes"]


class LabelEncoder:
    """Encodes and decodes action labels to/from integer indices."""

    def __init__(self, label_map_path: Optional[Path] = None):
        self._move_map: dict[str, int] = {}
        self._turn_map: dict[str, int] = {}
        self._jump_map: dict[str, int] = {}
        self._crouch_map: dict[str, int] = {}
        self._fire_map: dict[str, int] = {}
        self._macro_map: dict[str, int] = {}

        self._move_inv: dict[int, str] = {}
        self._turn_inv: dict[int, str] = {}
        self._macro_inv: dict[int, str] = {}

        if label_map_path is not None and label_map_path.exists():
            self._load_from_file(label_map_path)
        else:
            self._set_defaults()

    def _set_defaults(self) -> None:
        self._move_map = {label: idx for idx, label in enumerate(ACTION_MOVE_LABELS)}
        self._turn_map = {label: idx for idx, label in enumerate(ACTION_TURN_LABELS)}
        self._jump_map = {label: idx for idx, label in enumerate(JUMP_LABELS)}
        self._crouch_map = {label: idx for idx, label in enumerate(CROUCH_LABELS)}
        self._fire_map = {label: idx for idx, label in enumerate(FIRE_LABELS)}
        self._macro_map = {}  # populated from data

        self._move_inv = {v: k for k, v in self._move_map.items()}
        self._turn_inv = {v: k for k, v in self._turn_map.items()}

    def _load_from_file(self, path: Path) -> None:
        with open(path, "r") as f:
            data = json.load(f)

        if "action_move" in data:
            self._move_map = data["action_move"]
            self._move_inv = {v: k for k, v in self._move_map.items()}
        if "action_turn" in data:
            self._turn_map = data["action_turn"]
            self._turn_inv = {v: k for k, v in self._turn_map.items()}
        if "action_jump" in data:
            self._jump_map = data["action_jump"]
        if "action_crouch" in data:
            self._crouch_map = data["action_crouch"]
        if "action_fire" in data:
            self._fire_map = data["action_fire"]
        if "action_macro" in data:
            self._macro_map = data["action_macro"]

    @property
    def n_move_classes(self) -> int:
        return len(self._move_map)

    @property
    def n_turn_classes(self) -> int:
        return len(self._turn_map)

    def encode_move(self, label: str) -> int:
        return self._move_map.get(label, 0)

    def decode_move(self, idx: int) -> str:
        return self._move_inv.get(idx, "stop")

    def encode_turn(self, label: str) -> int:
        return self._turn_map.get(label, 0)

    def decode_turn(self, idx: int) -> str:
        return self._turn_inv.get(idx, "no_turn")

    def encode_jump(self, label: str) -> int:
        return self._jump_map.get(label, 0)

    def encode_crouch(self, label: str) -> int:
        return self._crouch_map.get(label, 0)

    def encode_fire(self, label: str) -> int:
        return self._fire_map.get(label, 0)

    def encode_macro(self, label: str) -> int:
        return self._macro_map.get(label, 0)

    def get_all_move_labels(self) -> list[str]:
        return sorted(self._move_map.keys(), key=lambda x: self._move_map[x])

    def get_all_turn_labels(self) -> list[str]:
        return sorted(self._turn_map.keys(), key=lambda x: self._turn_map[x])
