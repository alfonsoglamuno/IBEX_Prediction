import logging
import sys
from pathlib import Path
from src.utils.config import get, root


def _ensure_utf8_stdout() -> None:
    """On Windows the default console codec is cp1252 which can't encode many
    Unicode characters used in log messages. Reconfigure stdout/stderr to
    UTF-8 so box-drawing characters and arrows don't crash the process."""
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass  # Fail silently — worst case is garbled chars, not a crash


def get_logger(name: str) -> logging.Logger:
    _ensure_utf8_stdout()

    level   = getattr(logging, get("logging.level", "INFO"))
    log_dir = root() / get("logging.dir", "logs")
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_dir / "ibex.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
