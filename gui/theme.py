"""Theme primitives for the (legacy) GUI.

The test suite only requires that these objects are importable and instantiable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    """Base theme definition."""

    name: str = "Default"
    primary_color: str = "#1f2937"  # slate-800
    accent_color: str = "#3b82f6"  # blue-500
    background_color: str = "#ffffff"


@dataclass(frozen=True)
class ModernTheme(Theme):
    """A slightly more opinionated default theme."""

    name: str = "Modern"
    background_color: str = "#0b1220"  # dark
    primary_color: str = "#e5e7eb"  # gray-200
    accent_color: str = "#22c55e"  # green-500
