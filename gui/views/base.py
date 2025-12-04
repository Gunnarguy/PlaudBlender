import tkinter as tk
from tkinter import ttk
from typing import Dict, Callable

from gui.state import state


class BaseView(ttk.Frame):
    """Base frame for all views, with handy references to state & actions."""

    def __init__(self, parent: tk.Widget, actions: Dict[str, Callable]):
        super().__init__(parent, style="Main.TFrame", padding=8)
        self.actions = actions
        self.state = state
        self._build()

    # Template methods -------------------------------------------------
    def _build(self):  # pragma: no cover - UI
        """Called once on initialization."""

    def on_show(self):  # pragma: no cover - UI
        """Optional hook when view becomes visible."""

    # Convenience wrappers --------------------------------------------
    def call(self, name: str, *args, **kwargs):
        action = self.actions.get(name)
        if action:
            return action(*args, **kwargs)
        raise ValueError(f"Action '{name}' not registered")
