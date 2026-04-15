"""
Rich-based status overlay for the recorder.
Displays real-time recording status in the console.
"""

from __future__ import annotations

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.text import Text


class StatusOverlay:
    """Live status display for the recorder using Rich."""

    def __init__(self):
        self._console = Console()
        self._live: Live | None = None
        self._recording = False
        self._session_id = ""
        self._frame_count = 0
        self._event_count = 0
        self._fps_actual = 0.0
        self._dropped_frames = 0
        self._queue_size = 0
        self._paused = False
        self._scenario = ""
        self._elapsed_s = 0.0

    def update(
        self,
        recording: bool = False,
        session_id: str = "",
        frame_count: int = 0,
        event_count: int = 0,
        fps_actual: float = 0.0,
        dropped_frames: int = 0,
        queue_size: int = 0,
        paused: bool = False,
        scenario: str = "",
        elapsed_s: float = 0.0,
    ) -> None:
        """Update the status display."""
        self._recording = recording
        self._session_id = session_id
        self._frame_count = frame_count
        self._event_count = event_count
        self._fps_actual = fps_actual
        self._dropped_frames = dropped_frames
        self._queue_size = queue_size
        self._paused = paused
        self._scenario = scenario
        self._elapsed_s = elapsed_s

        if self._live is not None:
            self._live.update(self._build_table())

    def start(self) -> None:
        """Start the live display."""
        self._live = Live(self._build_table(), console=self._console, refresh_per_second=4)
        self._live.start()

    def stop(self) -> None:
        """Stop the live display."""
        if self._live is not None:
            self._live.stop()
            self._live = None

    def _build_table(self) -> Panel:
        table = Table(show_header=False, box=None, padding=(0, 1))

        # Status line
        status_color = "green" if self._recording else "red"
        status_text = "PAUSED" if self._paused else "RECORDING" if self._recording else "IDLE"
        table.add_row(
            Text("STATUS", style="bold"),
            Text(status_text, style=f"bold {status_color}"),
        )

        if self._session_id:
            table.add_row("Session", Text(self._session_id, style="cyan"))

        if self._scenario:
            table.add_row("Scenario", Text(self._scenario, style="yellow"))

        table.add_row("Frames", str(self._frame_count))
        table.add_row("Events", str(self._event_count))
        table.add_row("FPS", f"{self._fps_actual:.1f}")
        table.add_row("Dropped", Text(str(self._dropped_frames), style="red" if self._dropped_frames > 0 else ""))
        table.add_row("Queue", Text(str(self._queue_size), style="yellow" if self._queue_size > 200 else ""))

        mins = int(self._elapsed_s // 60)
        secs = int(self._elapsed_s % 60)
        table.add_row("Elapsed", f"{mins:02d}:{secs:02d}")

        return Panel(table, title="[bold]CS2 Nav Dataset Recorder[/bold]", border_style=status_color)
