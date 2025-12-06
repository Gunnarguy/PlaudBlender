import json
import os
from typing import Dict, List

SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data", "saved_searches.json")


def _ensure_dir():
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)


def load_saved_searches() -> Dict[str, str]:
    _ensure_dir()
    if not os.path.exists(SAVE_PATH):
        return {}
    try:
        with open(SAVE_PATH, "r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        return {}
    return {}


def save_search(name: str, query: str) -> None:
    _ensure_dir()
    data = load_saved_searches()
    data[name] = query
    with open(SAVE_PATH, "w") as f:
        json.dump(data, f, indent=2)


def delete_search(name: str) -> None:
    data = load_saved_searches()
    if name in data:
        del data[name]
        with open(SAVE_PATH, "w") as f:
            json.dump(data, f, indent=2)


def list_saved_names() -> List[str]:
    return sorted(load_saved_searches().keys())
