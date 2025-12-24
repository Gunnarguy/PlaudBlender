"""Conflict detection (stub).

In a full system this would compare extracted facts/events to detect
contradictions. For tests we only provide an importable class and a `detect`
method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ConflictDetector:
    def detect(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # No-op stub: return empty conflict list.
        return []
