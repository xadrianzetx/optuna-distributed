from __future__ import annotations

import logging

from rich.logging import RichHandler


_default_handler: logging.Handler | None = None


def _get_library_logger() -> logging.Logger:
    library_name = __name__.split(".")[0]
    return logging.getLogger(library_name)


def _setup_logger() -> None:
    global _default_handler
    _default_handler = RichHandler(show_path=False)
    fmt = logging.Formatter(fmt="%(message)s", datefmt="[%X]")
    _default_handler.setFormatter(fmt)
    library_root_logger = _get_library_logger()
    library_root_logger.addHandler(_default_handler)
    library_root_logger.setLevel(logging.INFO)
    library_root_logger.propagate = False


_setup_logger()


def disable_logging() -> None:
    """Disables library level logger."""
    assert _default_handler is not None
    _get_library_logger().removeHandler(_default_handler)


def enable_logging() -> None:
    """Enables library level logger."""
    assert _default_handler is not None
    _get_library_logger().addHandler(_default_handler)
