"""Application state container.

This is a small, import-safe state object used by the GUI layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AppState:
    """Holds ephemeral UI state.

    The implementation is intentionally minimal for tests.
    """

    current_view: str = "dashboard"
    selected_recording_id: Optional[str] = None
    flags: Dict[str, Any] = field(default_factory=dict)
