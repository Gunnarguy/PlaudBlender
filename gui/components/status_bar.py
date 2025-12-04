import tkinter as tk
from tkinter import ttk
from gui.state import state


class StatusBar(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, style="Panel.TFrame", padding=(6, 3))
        self.message_var = tk.StringVar(value=state.status_message)
        self.busy_var = tk.StringVar(value="")

        ttk.Label(self, textvariable=self.message_var, style="Muted.TLabel").pack(side=tk.LEFT)
        ttk.Label(self, textvariable=self.busy_var, style="Muted.TLabel").pack(side=tk.RIGHT)

    def update_status(self):
        self.message_var.set(state.status_message)
        self.busy_var.set("●" if state.is_busy else "")
