import tkinter as tk
from tkinter import ttk
from gui.views.base import BaseView
from gui.state import state


class LogsView(BaseView):
    def _build(self):
        hero = ttk.Frame(self, style="Main.TFrame")
        hero.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(hero, text="Logs", style="Header.TLabel").pack(anchor='w')
        ttk.Label(hero, text="Runtime diagnostics (latest 500 entries)", style="Muted.TLabel").pack(anchor='w')

        control = ttk.Frame(self, style="Main.TFrame")
        control.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(control, text="Filter", style="Muted.TLabel").pack(side=tk.LEFT)
        self.filter_var = tk.StringVar()
        ttk.Entry(control, textvariable=self.filter_var, width=32).pack(side=tk.LEFT, padx=6)
        ttk.Button(control, text="Apply", command=self.refresh).pack(side=tk.LEFT)
        ttk.Button(control, text="Clear", command=self._clear_filter).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(control, text="Copy visible", command=self._copy).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(control, text="Save visible", command=self._save_visible).pack(side=tk.LEFT, padx=(6, 0))

        self.count_label = ttk.Label(control, text="0 lines", style="Muted.TLabel")
        self.count_label.pack(side=tk.RIGHT)

        log_frame = ttk.LabelFrame(self, text="Log output", padding=6, style="Panel.TLabelframe")
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.text = tk.Text(
            log_frame,
            bg="#0b1120",
            fg="#38bdf8",
            insertbackground="#38bdf8",
            font=("JetBrains Mono", 9),
            wrap=tk.NONE,
        )
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(log_frame, command=self.text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.configure(yscrollcommand=scrollbar.set)

        ttk.Button(self, text="Clear", command=self._clear).pack(anchor='e', pady=(6, 0))

    def refresh(self):
        self.text.delete('1.0', tk.END)
        logs = state.logs[-500:]
        term = (self.filter_var.get() or "").lower()
        if term:
            logs = [line for line in logs if term in line.lower()]
        self.text.insert('1.0', "\n".join(logs))
        self.count_label.config(text=f"{len(logs)} lines")

    def on_show(self):
        self.refresh()

    def _clear(self):
        state.logs.clear()
        self.refresh()

    def _copy(self):
        """Copy the currently visible log view to the clipboard."""
        try:
            content = self.text.get('1.0', tk.END)
            self.clipboard_clear()
            self.clipboard_append(content)
        except Exception:
            pass

    def _save_visible(self):
        """Save currently visible log view to a file."""
        try:
            from tkinter import filedialog

            content = self.text.get('1.0', tk.END)
            if not content.strip():
                return
            path = filedialog.asksaveasfilename(defaultextension=".log", filetypes=[("Log", "*.log"), ("Text", "*.txt"), ("All", "*.*")])
            if path:
                with open(path, 'w') as f:
                    f.write(content)
        except Exception:
            pass

    def _clear_filter(self):
        self.filter_var.set("")
        self.refresh()
