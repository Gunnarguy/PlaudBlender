import tkinter as tk
from tkinter import ttk
from gui.views.base import BaseView
from gui.state import state


class LogsView(BaseView):
    def _build(self):
        ttk.Label(self, text="Logs", style="Header.TLabel").pack(anchor='w', pady=(0, 4))
        self.text = tk.Text(self, bg="#0b1120", fg="#38bdf8", insertbackground="#38bdf8", font=("JetBrains Mono", 9))
        self.text.pack(fill=tk.BOTH, expand=True)
        ttk.Button(self, text="Clear", command=self._clear).pack(anchor='e', pady=(4, 0))

    def refresh(self):
        self.text.delete('1.0', tk.END)
        self.text.insert('1.0', "\n".join(state.logs[-500:]))

    def on_show(self):
        self.refresh()

    def _clear(self):
        state.logs.clear()
        self.refresh()
