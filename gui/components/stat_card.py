"""Stat card component (import-safe stub)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StatCard:
    label: str = ""
    value: str = ""
    helper_text: str = ""
