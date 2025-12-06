import tkinter as tk


class ToolTip:
    """Lightweight tooltip for Tk widgets.

    Usage:
        ToolTip(widget, "Helpful text")
    """

    def __init__(self, widget: tk.Widget, text: str, delay: int = 350):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tipwindow = None
        self._after_id = None
        self.widget.bind("<Enter>", self._schedule)
        self.widget.bind("<Leave>", self._hide)
        self.widget.bind("<ButtonPress>", self._hide)

    def _schedule(self, _event=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self):
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + 10
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#111827",
            foreground="#f1f5f9",
            relief=tk.SOLID,
            borderwidth=1,
            font=("Inter", 9),
            padx=6,
            pady=4,
        )
        label.pack()

    def _hide(self, _event=None):
        self._cancel()
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()
