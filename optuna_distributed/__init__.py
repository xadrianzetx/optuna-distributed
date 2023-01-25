from optuna_distributed.study import from_study


def _setup_logger() -> None:
    import logging

    from rich.logging import RichHandler

    handler = RichHandler(show_path=False)
    fmt = logging.Formatter(fmt="%(message)s", datefmt="[%X]")
    handler.setFormatter(fmt)
    root_logger = logging.getLogger(__name__)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)


_setup_logger()
__version__ = "0.4.0"
__all__ = ["from_study"]
