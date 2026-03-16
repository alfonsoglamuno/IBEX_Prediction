import logging
import sys
from pathlib import Path
from src.utils.config import get, root


def get_logger(name: str) -> logging.Logger:
    level = getattr(logging, get("logging.level", "INFO"))
    log_dir = root() / get("logging.dir", "logs")
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_dir / "ibex.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
