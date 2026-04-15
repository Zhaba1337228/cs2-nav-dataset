"""
Input logging module using pynput.
Captures keyboard and mouse events with monotonic timestamps.
"""

from __future__ import annotations

import logging
import threading
import time
from queue import Queue
from typing import Callable, Optional

from pynput import keyboard, mouse

from common.schemas import InputEvent

logger = logging.getLogger("cs2_nav.input_logger")

# Key name mapping — normalize pynput key names to simple strings
_KEY_MAP = {
    keyboard.Key.shift: "shift",
    keyboard.Key.shift_l: "shift",
    keyboard.Key.shift_r: "shift",
    keyboard.Key.ctrl: "ctrl",
    keyboard.Key.ctrl_l: "ctrl",
    keyboard.Key.ctrl_r: "ctrl",
    keyboard.Key.alt: "alt",
    keyboard.Key.alt_l: "alt",
    keyboard.Key.alt_r: "alt",
    keyboard.Key.space: "space",
    keyboard.Key.tab: "tab",
    keyboard.Key.enter: "enter",
    keyboard.Key.esc: "esc",
    keyboard.Key.backspace: "backspace",
    keyboard.Key.up: "up",
    keyboard.Key.down: "down",
    keyboard.Key.left: "left",
    keyboard.Key.right: "right",
}

# Mouse button mapping
_MOUSE_MAP = {
    mouse.Button.left: "left",
    mouse.Button.right: "right",
    mouse.Button.middle: "middle",
}


class InputLogger:
    """Logs keyboard and mouse events with timestamps."""

    def __init__(self):
        self._event_callback: Optional[Callable[[InputEvent], None]] = None
        self._kb_listener: Optional[keyboard.Listener] = None
        self._ms_listener: Optional[mouse.Listener] = None
        self._running = False
        self._lock = threading.Lock()

        # Track current key states
        self._key_states: dict[str, bool] = {}
        self._mouse_buttons: dict[str, bool] = {}
        self._last_mouse_dx: float = 0.0
        self._last_mouse_dy: float = 0.0

    def set_callback(self, callback: Callable[[InputEvent], None]) -> None:
        self._event_callback = callback

    def get_key_states(self) -> dict[str, bool]:
        """Return a snapshot of current key states."""
        with self._lock:
            return dict(self._key_states)

    def get_mouse_button_states(self) -> dict[str, bool]:
        """Return a snapshot of current mouse button states."""
        with self._lock:
            return dict(self._mouse_buttons)

    def _normalize_key(self, key) -> str:
        """Convert a pynput key to a simple string name."""
        if key in _KEY_MAP:
            return _KEY_MAP[key]
        if isinstance(key, keyboard.KeyCode):
            if key.char is not None:
                return key.char.lower()
            return f"key_{key.vk}"
        return str(key).replace("Key.", "")

    def _on_key_down(self, key) -> None:
        name = self._normalize_key(key)
        with self._lock:
            self._key_states[name] = True
        event = InputEvent(
            timestamp=time.monotonic(),
            event_type="key_down",
            key=name,
        )
        if self._event_callback:
            self._event_callback(event)

    def _on_key_up(self, key) -> None:
        name = self._normalize_key(key)
        with self._lock:
            self._key_states[name] = False
        event = InputEvent(
            timestamp=time.monotonic(),
            event_type="key_up",
            key=name,
        )
        if self._event_callback:
            self._event_callback(event)

    def _on_mouse_move(self, x: int, y: int, dx: int, dy: int) -> None:
        with self._lock:
            self._last_mouse_dx = float(dx)
            self._last_mouse_dy = float(dy)
        event = InputEvent(
            timestamp=time.monotonic(),
            event_type="mouse_move",
            dx=float(dx),
            dy=float(dy),
        )
        if self._event_callback:
            self._event_callback(event)

    def _on_mouse_click(self, x: int, y: int, button, pressed: bool) -> None:
        btn_name = _MOUSE_MAP.get(button, str(button))
        with self._lock:
            self._mouse_buttons[btn_name] = pressed
        event = InputEvent(
            timestamp=time.monotonic(),
            event_type="mouse_button_down" if pressed else "mouse_button_up",
            button=btn_name,
        )
        if self._event_callback:
            self._event_callback(event)

    def _on_mouse_scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        event = InputEvent(
            timestamp=time.monotonic(),
            event_type="mouse_move",
            dx=float(dx),
            dy=float(dy),
        )
        if self._event_callback:
            self._event_callback(event)

    def start(self) -> None:
        """Start listening for input events."""
        if self._running:
            return

        self._running = True
        self._kb_listener = keyboard.Listener(
            on_press=self._on_key_down,
            on_release=self._on_key_up,
            suppress=False,
        )
        self._ms_listener = mouse.Listener(
            on_move=self._on_mouse_move,
            on_click=self._on_mouse_click,
            on_scroll=self._on_mouse_scroll,
        )

        self._kb_listener.start()
        self._ms_listener.start()
        logger.info("Input listeners started")

    def stop(self) -> None:
        """Stop listening for input events."""
        self._running = False
        if self._kb_listener:
            self._kb_listener.stop()
            self._kb_listener = None
        if self._ms_listener:
            self._ms_listener.stop()
            self._ms_listener = None
        logger.info("Input listeners stopped")

    def is_running(self) -> bool:
        return self._running
