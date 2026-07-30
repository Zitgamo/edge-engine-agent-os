import logging
import sys

from src.config import Config


def setup_logging() -> None:
    config = Config()
    config.ensure_dirs()

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    root = logging.getLogger()
    root.setLevel(config.log_level.upper())

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if config.log_file:
        handlers.append(logging.FileHandler(config.log_file))

    for h in handlers:
        h.setFormatter(logging.Formatter(fmt))

    # Replace any pre-existing handlers to avoid double-logging
    root.handlers = handlers
