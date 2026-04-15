"""
Logging setup with Rich for console output and file logging.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


_console = Console()


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    module: str = "cs2_nav",
) -> logging.Logger:
    """Configure and return a logger with Rich console handler and optional file handler."""
    logger = logging.getLogger(module)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Rich console handler
    rich_handler = RichHandler(
        console=_console,
        show_path=False,
        show_time=True,
        markup=True,
        rich_tracebacks=True,
    )
    rich_handler.setLevel(level)
    logger.addHandler(rich_handler)

    # Optional file handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "cs2_nav") -> logging.Logger:
    """Get a child logger."""
    return logging.getLogger(name).getChild(name.split(".")[-1] if "." in name else name)
