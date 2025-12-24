"""Base class for GUI views.

The actual UI framework is not required for tests; this is a structural stub.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaseView:
    name: str = "base"
