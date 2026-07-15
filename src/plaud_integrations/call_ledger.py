"""Append-only, redacted call ledger shared by all PLAUD transports."""

from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any

from .models import PlaudCallEvent
from .redaction import redact


class PlaudCallLedger:
    def __init__(self, path: str | Path | None = None):
        root = Path(__file__).resolve().parents[2]
        self.path = Path(path) if path else root / "data" / "plaud-call-ledger.jsonl"
        self._lock = threading.Lock()

    def record(self, event: PlaudCallEvent) -> None:
        payload = redact(event.to_dict())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-max(0, limit) :]
        return [json.loads(line) for line in lines if line.strip()]


default_ledger = PlaudCallLedger()
