"""
Global configuration for the CS2 navigation dataset pipeline.
All tunable parameters live here as dataclasses with sensible defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────
# Project root & dataset paths
# ─────────────────────────────────────────────

@dataclass
class PathConfig:
    """Directory layout for raw and processed data."""
    project_root: Path = Path(__file__).resolve().parent
    dataset_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "dataset")
    raw_sessions_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "dataset" / "raw_sessions")
    processed_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "dataset" / "processed")
    manifests_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "dataset" / "manifests")
    debug_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "dataset" / "debug")

    def ensure_dirs(self) -> None:
        for d in [
            self.dataset_dir,
            self.raw_sessions_dir,
            self.processed_dir,
            self.manifests_dir,
            self.debug_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Capture / recorder settings
# ─────────────────────────────────────────────

@dataclass
class CaptureConfig:
    """Screen capture parameters."""
    fps: int = 15                          # target frames per second
    target_width: int = 640                # resize width before saving
    target_height: int = 480               # resize height before saving
    jpeg_quality: int = 95                 # JPEG save quality
    window_title: str = "Counter-Strike 2" # window title to capture (empty = full screen)
    monitor_index: int = 0                 # which monitor (0-based)
    use_region: bool = False               # capture a sub-region instead of full screen
    region_x: int = 0
    region_y: int = 0
    region_width: int = 1920
    region_height: int = 1080


@dataclass
class RecorderConfig:
    """High-level recorder behaviour."""
    max_queue_size: int = 500              # max frames in writer queue
    flush_on_stop_timeout_s: float = 30.0  # how long to wait for queue drain
    frame_filename_fmt: str = "{:06d}.jpg" # frame file naming pattern
    session_prefix: str = "session"        # session directory naming


# ─────────────────────────────────────────────
# Labeling thresholds
# ─────────────────────────────────────────────

@dataclass
class LabelingConfig:
    """Thresholds for converting raw input into action labels."""
    # Mouse movement thresholds (pixels per frame interval)
    mouse_small_threshold: float = 5.0
    mouse_medium_threshold: float = 30.0
    mouse_large_threshold: float = 80.0

    # Minimum duration (ms) for an action to be considered intentional
    min_action_duration_ms: float = 50.0

    # Window (ms) for detecting "stuck" behaviour (keys pressed but little mouse movement)
    unstuck_detection_window_ms: float = 2000.0

    # Epsilon for considering mouse "idle" (abs(dx)+abs(dy) < epsilon)
    idle_mouse_epsilon: float = 3.0

    # Key combination weights for macro action naming
    diagonal_blend_threshold: float = 0.3  # if both W and A/D pressed, classify as diagonal


# ─────────────────────────────────────────────
# Export / dataset settings
# ─────────────────────────────────────────────

@dataclass
class ExportConfig:
    """Dataset export parameters."""
    val_split_ratio: float = 0.15          # fraction of samples for validation
    random_seed: int = 42                  # reproducibility seed for splitting
    history_len: int = 1                   # sequence length for temporal samples
    image_size: tuple[int, int] = (224, 224)  # default image size for training


# ─────────────────────────────────────────────
# Master config
# ─────────────────────────────────────────────

@dataclass
class Config:
    """Top-level configuration aggregating all sub-configs."""
    paths: PathConfig = field(default_factory=PathConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    recorder: RecorderConfig = field(default_factory=RecorderConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    def to_dict(self) -> dict:
        return asdict(self)
