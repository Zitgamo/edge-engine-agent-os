import logging
import sys

from src.config import Config


def setup_logging() -> None:
    config = Config()
    config.ensure_dirs()

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
    ]

    if config.log_file:
        handlers.append(logging.FileHandler(config.log_file))

    logging.basicConfig(
        level=config.log_level.upper(),
        format=fmt,
        handlers=handlers,
    )
