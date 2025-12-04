from tkinter import ttk


class StatCard(ttk.Frame):
    def __init__(self, parent, title: str, icon: str = "", value: str = "—"):
        super().__init__(parent, style="Card.TFrame", padding=(8, 6))
        full_title = f"{icon} {title}" if icon else title
        self.title = ttk.Label(self, text=full_title, style="CardTitle.TLabel")
        self.title.pack(anchor="w")
        self.value_var = ttk.Label(self, text=value, style="CardValue.TLabel")
        self.value_var.pack(anchor="w", pady=(4, 0))

    def update_value(self, value):
        self.value_var.configure(text=str(value))
