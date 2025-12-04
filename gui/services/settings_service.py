import os
from pathlib import Path
from dotenv import set_key

from gui.utils.logging import log

ENV_PATH = Path(__file__).resolve().parents[2] / '.env'


def save_settings(values: dict):
    for key, value in values.items():
        os.environ[key] = value
        if ENV_PATH.exists():
            set_key(str(ENV_PATH), key, value)
    log('INFO', f"Saved {len(values)} settings to .env")
