"""
Async disk writer.
Writes frames and events to disk from a queue to avoid blocking the capture loop.
"""

from __future__ import annotations

import csv
import logging
import threading
import time
from pathlib import Path
from queue import Queue, Empty
from typing import Optional

import cv2
import numpy as np

from common.schemas import InputEvent

logger = logging.getLogger("cs2_nav.writer")


class DiskWriter:
    """Async writer that consumes frames and events from queues and writes to disk."""

    def __init__(self, max_queue_size: int = 500):
        self._frame_queue: Queue[tuple[np.ndarray, Path]] = Queue(maxsize=max_queue_size)
        self._event_queue: Queue[InputEvent] = Queue(maxsize=max_queue_size * 4)
        self._events_file: Optional[object] = None
        self._events_writer: Optional[csv.writer] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._max_queue_size = max_queue_size

    @property
    def frame_queue_size(self) -> int:
        return self._frame_queue.qsize()

    @property
    def event_queue_size(self) -> int:
        return self._event_queue.qsize()

    def open_events_file(self, path: Path) -> None:
        """Open the events CSV file for writing."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self._events_file = open(path, "w", newline="", encoding="utf-8")
        self._events_writer = csv.writer(self._events_file)
        # Write header
        self._events_writer.writerow(["timestamp", "event_type", "key", "button", "dx", "dy"])

    def enqueue_frame(self, frame: np.ndarray, path: Path) -> bool:
        """Enqueue a frame for writing. Returns False if queue is full."""
        try:
            self._frame_queue.put_nowait((frame, path))
            return True
        except Exception:
            return False

    def enqueue_event(self, event: InputEvent) -> bool:
        """Enqueue an event for writing. Returns False if queue is full."""
        try:
            self._event_queue.put_nowait(event)
            return True
        except Exception:
            return False

    def start(self) -> None:
        """Start the writer worker thread."""
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker, daemon=True, name="disk_writer")
        self._worker_thread.start()
        logger.info("Disk writer thread started")

    def stop(self, timeout: float = 30.0) -> None:
        """Stop the writer and wait for queue to drain."""
        self._running = False
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=timeout)
        self._flush_events()
        self._close_events_file()
        logger.info("Disk writer stopped")

    def _worker(self) -> None:
        """Worker loop: drain queues and write to disk."""
        while self._running:
            wrote_something = False

            # Drain frames
            for _ in range(50):  # batch up to 50 frames per iteration
                try:
                    frame, path = self._frame_queue.get_nowait()
                    self._write_frame(frame, path)
                    self._frame_queue.task_done()
                    wrote_something = True
                except Empty:
                    break

            # Drain events
            for _ in range(200):  # batch up to 200 events per iteration
                try:
                    event = self._event_queue.get_nowait()
                    self._write_event(event)
                    self._event_queue.task_done()
                    wrote_something = True
                except Empty:
                    break

            if not wrote_something:
                time.sleep(0.005)  # 5ms sleep when idle

        # Final drain
        self._drain_frames()
        self._drain_events()

    def _drain_frames(self) -> None:
        while True:
            try:
                frame, path = self._frame_queue.get_nowait()
                self._write_frame(frame, path)
                self._frame_queue.task_done()
            except Empty:
                break

    def _drain_events(self) -> None:
        while True:
            try:
                event = self._event_queue.get_nowait()
                self._write_event(event)
                self._event_queue.task_done()
            except Empty:
                break

    def _write_frame(self, frame: np.ndarray, path: Path) -> None:
        """Write a single frame to disk."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), frame)
        except Exception as e:
            logger.error(f"Failed to write frame {path}: {e}")

    def _write_event(self, event: InputEvent) -> None:
        """Write a single event to the CSV file."""
        if self._events_writer is None:
            return
        try:
            self._events_writer.writerow(event.to_csv_row())
        except Exception as e:
            logger.error(f"Failed to write event: {e}")

    def _flush_events(self) -> None:
        if self._events_file is not None:
            try:
                self._events_file.flush()
            except Exception:
                pass

    def _close_events_file(self) -> None:
        if self._events_file is not None:
            try:
                self._events_file.close()
            except Exception:
                pass
            self._events_file = None
            self._events_writer = None
