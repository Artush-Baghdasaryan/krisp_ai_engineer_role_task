import logging
import os
import sys


def setup_logging(level: int | None = None) -> None:
    if level is None:
        level_name = (os.getenv("LOG_LEVEL") or "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)

    fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)

    for name in ("httpx", "httpcore", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)
