"""Logging setup. I use one named logger ("task_agent") everywhere and it
writes to the console and to logs/agent.log. Every state change, tool call and
request goes through it, so the log file is also my evidence for the report."""

import logging
from pathlib import Path

from app import config

LOGGER_NAME = "task_agent"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: int = logging.INFO, log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    # I only want to add the handlers once, otherwise every line prints twice
    if getattr(logger, "_configured", False):
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_file = log_file or config.LOG_FILE
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:
        # logging problems shouldn't take the whole app down
        logger.warning("Could not open log file %s: %s", log_file, exc)

    logger._configured = True  # type: ignore[attr-defined]
    return logger


def get_logger(child: str | None = None) -> logging.Logger:
    base = logging.getLogger(LOGGER_NAME)
    return base.getChild(child) if child else base
