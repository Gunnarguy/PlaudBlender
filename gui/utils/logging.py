import logging
from datetime import datetime
from gui.state import state

logger = logging.getLogger("PlaudBlender")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def log(level: str, message: str):
    level = level.upper()
    getattr(logger, level.lower(), logger.info)(message)
    timestamp = datetime.now().strftime("%H:%M:%S")
    state.logs.append(f"[{timestamp}] {level}: {message}")
