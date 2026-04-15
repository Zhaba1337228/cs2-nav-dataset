"""
Screen capture module using mss.
Captures frames at a fixed FPS, optionally targeting a specific window or region.
"""

from __future__ import annotations

import logging
import time
from queue import Queue
from typing import Callable, Optional

import cv2
import mss
import numpy as np

from config import CaptureConfig

logger = logging.getLogger("cs2_nav.capture")


class ScreenCapture:
    """Captures screen frames at a target FPS using mss."""

    def __init__(self, cfg: Optional[CaptureConfig] = None):
        self.cfg = cfg or CaptureConfig()
        self._running = False
        self._sct: Optional[mss.mss] = None
        self._monitor: dict = {}

    def _init_monitor(self) -> None:
        """Initialize the mss monitor region."""
        self._sct = mss.mss()
        monitors = self._sct.monitors  # monitors[0] is full virtual screen, 1+ are individual
        idx = min(self.cfg.monitor_index + 1, len(monitors) - 1)
        if idx < 1:
            idx = 1
        self._monitor = monitors[idx]

        if self.cfg.use_region:
            self._monitor = {
                "left": self.cfg.region_x,
                "top": self.cfg.region_y,
                "width": self.cfg.region_width,
                "height": self.cfg.region_height,
            }
        logger.info(
            f"Capture region: {self._monitor['left']},{self._monitor['top']} "
            f"{self._monitor['width']}x{self._monitor['height']}"
        )

    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame. Returns BGR numpy array or None on error."""
        if self._sct is None:
            self._init_monitor()

        try:
            sct_img = self._sct.grab(self._monitor)
            # Convert to BGR (mss returns BGRA)
            frame = np.array(sct_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Resize to target dimensions
            if frame.shape[1] != self.cfg.target_width or frame.shape[0] != self.cfg.target_height:
                frame = cv2.resize(frame, (self.cfg.target_width, self.cfg.target_height))

            return frame
        except Exception as e:
            logger.error(f"Frame capture error: {e}")
            return None

    def run_loop(
        self,
        frame_callback: Callable[[np.ndarray, float], None],
        running_flag: Callable[[], bool],
    ) -> None:
        """
        Continuous capture loop. Calls frame_callback(frame, timestamp) for each frame.
        running_flag() should return False to stop.
        """
        self._init_monitor()
        self._running = True
        interval = 1.0 / self.cfg.fps
        frame_count = 0
        dropped = 0

        logger.info(f"Capture loop started at {self.cfg.fps} FPS")

        while self._running and running_flag():
            t_start = time.monotonic()

            frame = self.capture_frame()
            if frame is not None:
                timestamp = time.monotonic()
                frame_callback(frame, timestamp)
                frame_count += 1
            else:
                dropped += 1

            # Sleep to maintain target FPS
            elapsed = time.monotonic() - t_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Frame took longer than interval — dropped
                dropped += 1
                if dropped % 30 == 0:
                    logger.warning(f"Dropped {dropped} frames so far")

        self._running = False
        logger.info(f"Capture loop ended. Captured: {frame_count}, Dropped: {dropped}")

    def close(self) -> None:
        self._running = False
        if self._sct is not None:
            self._sct.close()
            self._sct = None
