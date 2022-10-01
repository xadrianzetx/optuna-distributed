from optuna_distributed.study import from_study


def _setup_logger() -> None:
    import logging

    import colorlog

    fmt = colorlog.ColoredFormatter(
        "%(log_color)s[%(levelname)1.1s %(asctime)s]%(reset)s %(message)s"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    root_logger = logging.getLogger(__name__)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)


_setup_logger()
__version__ = "0.1.0"
__all__ = ["from_study"]
