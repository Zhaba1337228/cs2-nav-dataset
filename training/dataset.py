"""
PyTorch Dataset for CS2 navigation imitation learning.

Supports:
- Single-frame and sequence (history) mode
- Manifest JSONL or parquet input
- Configurable image size and transforms
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

from training.label_maps import LabelEncoder


class NavigationDataset(Dataset):
    """
    PyTorch Dataset for CS2 navigation imitation learning.

    Args:
        manifest_path: Path to a JSONL manifest file (train_manifest.jsonl or val_manifest.jsonl).
        dataset_root: Root directory of the dataset (for resolving image paths).
        history_len: Number of consecutive frames per sample (1 = single frame).
        image_size: Target (height, width) for image resizing.
        transform: Optional torchvision-style transform to apply to images.
        label_encoder: Optional LabelEncoder instance.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        dataset_root: Optional[str | Path] = None,
        history_len: int = 1,
        image_size: tuple[int, int] = (224, 224),
        transform: Optional[Callable] = None,
        label_encoder: Optional[LabelEncoder] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.dataset_root = Path(dataset_root) if dataset_root else self.manifest_path.parent.parent
        self.history_len = history_len
        self.image_size = image_size  # (height, width)
        self.transform = transform
        self.label_encoder = label_encoder or LabelEncoder()

        # Load manifest
        self._records = self._load_manifest()

        # Group by session for sequence access
        self._session_indices: dict[str, list[int]] = {}
        for idx, rec in enumerate(self._records):
            sid = rec["session_id"]
            if sid not in self._session_indices:
                self._session_indices[sid] = []
            self._session_indices[sid].append(idx)

        # Sort each session's indices by frame_index
        for sid in self._session_indices:
            self._session_indices[sid].sort(
                key=lambda i: self._records[i].get("frame_index", 0)
            )

        # Build flat index mapping for __getitem__
        self._flat_indices: list[tuple[str, int]] = []  # (session_id, local_index)
        for sid, indices in self._session_indices.items():
            for local_idx in range(len(indices)):
                self._flat_indices.append((sid, local_idx))

    def _load_manifest(self) -> list[dict]:
        """Load records from a JSONL manifest file."""
        records = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def __len__(self) -> int:
        return len(self._flat_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        """
        Returns:
            images: Tensor of shape (history_len, C, H, W)
            target: Dict of action labels
            metadata: Dict of sample metadata
        """
        session_id, local_idx = self._flat_indices[idx]

        # Get the sequence of frame indices for this sample
        session_indices = self._session_indices[session_id]
        start = max(0, local_idx - self.history_len + 1)
        sequence_local = session_indices[start:local_idx + 1]

        # Pad if needed (repeat first frame)
        while len(sequence_local) < self.history_len:
            sequence_local.insert(0, sequence_local[0])

        # Load images
        images = []
        for li in sequence_local:
            global_idx = session_indices[li]
            rec = self._records[global_idx]
            image = self._load_image(rec["image_path"])
            images.append(image)

        # Stack to tensor: (history_len, C, H, W)
        images_tensor = torch.stack(images, dim=0)

        # Build target dict from the last frame in the sequence
        last_rec = self._records[session_indices[local_idx]]
        target = self._build_target(last_rec)

        # Metadata
        metadata = {
            "sample_id": last_rec.get("sample_id", ""),
            "session_id": session_id,
            "frame_index": last_rec.get("frame_index", 0),
            "map_name": last_rec.get("map_name", "unknown"),
            "scenario_type": last_rec.get("scenario_type", "navigation"),
        }

        return images_tensor, target, metadata

    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess a single image."""
        full_path = self.dataset_root / image_path
        if not full_path.exists():
            # Fallback: create a blank image
            img = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        else:
            img = np.array(Image.open(full_path).convert("RGB"))
            if not self.transform:
                # Only resize here if no transform is provided
                img = np.array(Image.fromarray(img).resize((self.image_size[1], self.image_size[0])))

        # Apply transforms if provided (transform should handle resizing)
        if self.transform is not None:
            img = np.array(self.transform(Image.fromarray(img)))

        # Convert to tensor: (C, H, W), float32, normalized to [0, 1]
        if isinstance(img, np.ndarray):
            tensor = torch.from_numpy(img).float()
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)
            elif tensor.dim() == 3 and tensor.shape[-1] == 3:
                tensor = tensor.permute(2, 0, 1)
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            return tensor
        return img

    def _build_target(self, rec: dict) -> dict[str, Any]:
        """Build the target label dict from a sample record."""
        return {
            "action_move": self.label_encoder.encode_move(rec.get("action_move", "stop")),
            "action_turn": self.label_encoder.encode_turn(rec.get("action_turn", "no_turn")),
            "action_jump": int(rec.get("action_jump", 0)),
            "action_crouch": int(rec.get("action_crouch", 0)),
            "action_fire": int(rec.get("action_fire", 0)),
            "action_macro": self.label_encoder.encode_macro(rec.get("action_macro", "idle")),
            # Also include raw key states as regression targets
            "key_w": int(rec.get("key_w", 0)),
            "key_a": int(rec.get("key_a", 0)),
            "key_s": int(rec.get("key_s", 0)),
            "key_d": int(rec.get("key_d", 0)),
            "mouse_dx": float(rec.get("mouse_dx", 0.0)),
            "mouse_dy": float(rec.get("mouse_dy", 0.0)),
        }

    @property
    def n_move_classes(self) -> int:
        return self.label_encoder.n_move_classes

    @property
    def n_turn_classes(self) -> int:
        return self.label_encoder.n_turn_classes

    def get_class_weights(self, target_key: str = "action_move") -> torch.Tensor:
        """Compute inverse-frequency class weights for the given target."""
        encoder = self.label_encoder
        encode_fn = {
            "action_move": encoder.encode_move,
            "action_turn": encoder.encode_turn,
            "action_jump": encoder.encode_jump,
            "action_crouch": encoder.encode_crouch,
            "action_fire": encoder.encode_fire,
            "action_macro": encoder.encode_macro,
        }.get(target_key, encoder.encode_move)

        n_classes_map = {
            "action_move": encoder.n_move_classes,
            "action_turn": encoder.n_turn_classes,
            "action_jump": 2,
            "action_crouch": 2,
            "action_fire": 2,
        }
        n_classes = n_classes_map.get(target_key, encoder.n_move_classes)

        counts = {}
        for rec in self._records:
            label = rec.get(target_key, "stop")
            idx = encode_fn(label)
            counts[idx] = counts.get(idx, 0) + 1

        total = sum(counts.values())
        weights = torch.zeros(n_classes)
        for idx, count in counts.items():
            weights[idx] = total / max(count, 1)
        weights = weights / weights.sum() * n_classes  # normalize
        return weights
