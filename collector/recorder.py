"""
Recorder orchestrator.
Coordinates screen capture, input logging, and disk writing into a single recording session.
"""

from __future__ import annotations

import json
import logging
import signal
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from config import Config
from collector.capture import ScreenCapture
from collector.input_logger import InputLogger
from collector.writer import DiskWriter
from collector.overlay import StatusOverlay
from common.schemas import InputEvent, SessionMeta
from common.paths import get_frames_dir, get_events_path, get_session_meta_path
from common.utils import now_iso, monotonic_s, generate_session_id, find_next_session_index

logger = logging.getLogger("cs2_nav.recorder")


class Recorder:
    """
    Main recorder that orchestrates capture, input logging, and disk writing.

    Usage:
        recorder = Recorder(cfg)
        recorder.start_session()
        # ... recording happens ...
        recorder.stop_session()
    """

    # Hotkey: F6 = start/stop, F7 = pause/resume, F8 = mark scenario, F9 = new session
    HOTKEY_STOP = "f6"
    HOTKEY_PAUSE = "f7"
    HOTKEY_SCENARIO = "f8"
    HOTKEY_NEW_SESSION = "f9"

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.cfg.paths.ensure_dirs()

        self._capture = ScreenCapture(self.cfg.capture)
        self._input_logger = InputLogger()
        self._writer = DiskWriter(max_queue_size=self.cfg.recorder.max_queue_size)
        self._overlay = StatusOverlay()

        self._running = False
        self._paused = False
        self._session_id = ""
        self._frame_count = 0
        self._event_count = 0
        self._dropped_frames = 0
        self._scenario_type = "navigation"
        self._map_name = "unknown"
        self._session_start_time = 0.0
        self._session_meta: Optional[SessionMeta] = None

        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # FPS tracking
        self._fps_window: list[float] = []
        self._fps_actual = 0.0

        # Graceful shutdown
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)

    def start_session(
        self,
        session_id: Optional[str] = None,
        map_name: str = "unknown",
        scenario_type: str = "navigation",
    ) -> str:
        """Start a new recording session."""
        # Determine session ID
        if session_id is None:
            idx = find_next_session_index(self.cfg.paths.raw_sessions_dir)
            session_id = generate_session_id(self.cfg.recorder.session_prefix, idx)

        self._session_id = session_id
        self._map_name = map_name
        self._scenario_type = scenario_type
        self._frame_count = 0
        self._event_count = 0
        self._dropped_frames = 0
        self._fps_window = []
        self._fps_actual = 0.0

        # Create session directories
        frames_dir = get_frames_dir(self._session_id, self.cfg)
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Open events file
        events_path = get_events_path(self._session_id, self.cfg)
        self._writer.open_events_file(events_path)

        # Create session metadata
        self._session_meta = SessionMeta(
            session_id=self._session_id,
            map_name=map_name,
            scenario_type=scenario_type,
            start_time=now_iso(),
            fps=self.cfg.capture.fps,
        )

        # Set up input event callback
        self._input_logger.set_callback(self._on_input_event)

        # Start components
        self._writer.start()
        self._input_logger.start()

        # Start capture thread
        self._running = True
        self._paused = False
        self._stop_event.clear()
        self._session_start_time = monotonic_s()

        self._capture_thread = threading.Thread(
            target=self._capture.run_loop,
            args=(self._on_frame, self._is_running),
            daemon=True,
            name="capture_loop",
        )
        self._capture_thread.start()

        # Install signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Start overlay
        self._overlay.start()

        logger.info(f"Session {self._session_id} started (map={map_name}, scenario={scenario_type})")
        return self._session_id

    def stop_session(self) -> str:
        """Stop the current recording session and save metadata."""
        self._running = False
        self._stop_event.set()

        # Stop overlay
        self._overlay.stop()

        # Wait for capture thread
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=10)
            self._capture_thread = None

        # Stop input logger
        self._input_logger.stop()

        # Stop writer (drains queues)
        self._writer.stop(timeout=self.cfg.recorder.flush_on_stop_timeout_s)

        # Save session metadata
        self._save_session_meta()

        # Restore signal handlers
        signal.signal(signal.SIGINT, self._original_sigint)
        signal.signal(signal.SIGTERM, self._original_sigterm)

        logger.info(
            f"Session {self._session_id} stopped. "
            f"Frames: {self._frame_count}, Events: {self._event_count}, "
            f"Dropped: {self._dropped_frames}"
        )
        return self._session_id

    def pause_recording(self) -> None:
        """Pause recording (input still logged but frames not captured)."""
        self._paused = not self._paused
        state = "paused" if self._paused else "resumed"
        logger.info(f"Recording {state}")

    def set_scenario(self, scenario: str) -> None:
        """Mark the current scenario type."""
        self._scenario_type = scenario
        if self._session_meta:
            self._session_meta.scenario_type = scenario
        logger.info(f"Scenario set to: {scenario}")

    def set_map(self, map_name: str) -> None:
        """Set the current map name."""
        self._map_name = map_name
        if self._session_meta:
            self._session_meta.map_name = map_name

    def is_running(self) -> bool:
        return self._running

    def is_paused(self) -> bool:
        return self._paused

    # ─────────────────────────────────────────────
    # Internal callbacks
    # ─────────────────────────────────────────────

    def _on_frame(self, frame: np.ndarray, timestamp: float) -> None:
        """Called by capture loop for each frame."""
        if self._paused:
            return

        frame_path = get_frame_path(self._session_id, self._frame_count, self.cfg)
        success = self._writer.enqueue_frame(frame, frame_path)
        if success:
            self._frame_count += 1
            # FPS tracking
            self._fps_window.append(timestamp)
            # Keep only last 2 seconds of timestamps
            cutoff = timestamp - 2.0
            self._fps_window = [t for t in self._fps_window if t > cutoff]
            if len(self._fps_window) >= 2:
                self._fps_actual = (len(self._fps_window) - 1) / (self._fps_window[-1] - self._fps_window[0])
        else:
            self._dropped_frames += 1

    def _on_input_event(self, event: InputEvent) -> None:
        """Called by input logger for each input event."""
        # Check for hotkeys
        if event.event_type == "key_down":
            if event.key == self.HOTKEY_STOP:
                logger.info("Hotkey: stop recording")
                self._stop_event.set()
                return
            elif event.key == self.HOTKEY_PAUSE:
                self.pause_recording()
                return
            elif event.key == self.HOTKEY_SCENARIO:
                # Cycle through common scenario types
                scenarios = ["navigation", "navigation_corridor", "navigation_doorway",
                             "obstacle_avoidance", "unstuck", "open_area"]
                current_idx = scenarios.index(self._scenario_type) if self._scenario_type in scenarios else 0
                next_scenario = scenarios[(current_idx + 1) % len(scenarios)]
                self.set_scenario(next_scenario)
                return
            elif event.key == self.HOTKEY_NEW_SESSION:
                logger.info("Hotkey: new session requested")
                # Stop current, start new
                self.stop_session()
                idx = find_next_session_index(self.cfg.paths.raw_sessions_dir)
                self.start_session(
                    session_id=generate_session_id(self.cfg.recorder.session_prefix, idx),
                    map_name=self._map_name,
                    scenario_type=self._scenario_type,
                )
                return

        self._event_count += 1
        self._writer.enqueue_event(event)

    def _is_running(self) -> bool:
        return self._running and not self._stop_event.is_set()

    def _save_session_meta(self) -> None:
        if self._session_meta is None:
            return
        self._session_meta.end_time = now_iso()
        self._session_meta.frame_count = self._frame_count
        self._session_meta.event_count = self._event_count
        meta_path = get_session_meta_path(self._session_id, self.cfg)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._session_meta.to_dict(), f, indent=2)

    def _signal_handler(self, signum, frame) -> None:
        logger.info(f"Signal {signum} received — stopping session gracefully")
        self._stop_event.set()

    def _update_overlay(self) -> None:
        elapsed = monotonic_s() - self._session_start_time if self._session_start_time > 0 else 0
        self._overlay.update(
            recording=self._running,
            session_id=self._session_id,
            frame_count=self._frame_count,
            event_count=self._event_count,
            fps_actual=self._fps_actual,
            dropped_frames=self._dropped_frames,
            queue_size=self._writer.frame_queue_size,
            paused=self._paused,
            scenario=self._scenario_type,
            elapsed_s=elapsed,
        )
