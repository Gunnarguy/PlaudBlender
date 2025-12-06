import tkinter as tk
from tkinter import ttk

class ModernTheme:
    """Centralized styling for the PlaudBlender desktop app."""

    COLORS = {
        "bg_dark": "#0f172a",
        "bg_main": "#111827",
        "bg_panel": "#1f2937",
        "bg_card": "#374151",
        "accent": "#38bdf8",
        "accent_muted": "#0ea5e9",
        "text_main": "#f8fafc",
        "text_muted": "#cbd5f5",
        "error": "#f87171",
        "success": "#34d399",
        "warning": "#facc15",
        "border": "#273449",
    }

    FONTS = {
        "h1": ("Inter", 16, "bold"),
        "h2": ("Inter", 13, "bold"),
        "h3": ("Inter", 11, "bold"),
        "body": ("Inter", 10),
        "body_sm": ("Inter", 9),
        "mono": ("JetBrains Mono", 9),
    }

    @classmethod
    def apply(cls, root):
        style = ttk.Style(root)
        style.theme_use("clam")
        colors = cls.COLORS

        root.configure(bg=colors["bg_dark"])

        style.configure("Sidebar.TFrame", background=colors["bg_dark"])
        style.configure("Main.TFrame", background=colors["bg_main"])
        style.configure("Panel.TFrame", background=colors["bg_panel"], borderwidth=0)
        style.configure("Card.TFrame", background=colors["bg_card"], relief="flat", borderwidth=0)
        style.configure("Panel.TLabelframe", background=colors["bg_panel"], borderwidth=0)
        style.configure("Panel.TLabelframe.Label", background=colors["bg_panel"], foreground=colors["text_muted"], font=cls.FONTS["body_sm"])

        style.configure("TLabel", background=colors["bg_main"], foreground=colors["text_main"], font=cls.FONTS["body"])
        style.configure("Sidebar.TLabel", background=colors["bg_dark"], foreground=colors["text_muted"], font=cls.FONTS["body"])
        style.configure("Header.TLabel", background=colors["bg_main"], foreground=colors["text_main"], font=cls.FONTS["h2"])
        style.configure("Muted.TLabel", background=colors["bg_main"], foreground=colors["text_muted"], font=cls.FONTS["body_sm"])
        style.configure("CardTitle.TLabel", background=colors["bg_card"], foreground=colors["text_muted"], font=cls.FONTS["body_sm"])
        style.configure("CardValue.TLabel", background=colors["bg_card"], foreground=colors["text_main"], font=cls.FONTS["h2"])

        # Pills / badges for inline status chips
        style.configure("Badge.TLabel", background=colors["bg_card"], foreground=colors["accent"],
                padding=(8, 2), borderwidth=1, relief="solid", font=cls.FONTS["body_sm"])
        style.configure("SuccessBadge.TLabel", background=colors["bg_card"], foreground=colors["success"],
                padding=(8, 2), borderwidth=1, relief="solid", font=cls.FONTS["body_sm"])
        style.configure("WarnBadge.TLabel", background=colors["bg_card"], foreground=colors["warning"],
                padding=(8, 2), borderwidth=1, relief="solid", font=cls.FONTS["body_sm"])

        # Toggleable pill buttons (used for filters)
        style.configure("Pill.TButton", background=colors["bg_panel"], foreground=colors["text_muted"],
                borderwidth=1, relief="solid", padding=(10, 4), font=cls.FONTS["body_sm"])
        style.map("Pill.TButton",
              background=[("active", colors["bg_card"])],
              foreground=[("active", colors["text_main"])])
        style.configure("PillActive.TButton", background=colors["accent"], foreground="black",
                borderwidth=0, padding=(10, 4), font=cls.FONTS["body_sm"])

        style.configure("TButton", background=colors["bg_card"], foreground=colors["text_main"], borderwidth=0, padding=(6, 3))
        style.map("TButton", background=[("active", colors["accent_muted"])])

        style.configure("Accent.TButton", background=colors["accent"], foreground="black", font=cls.FONTS["body"], padding=(8, 4))
        style.map("Accent.TButton", background=[("active", colors["accent_muted"])])

        style.configure("Nav.TButton", background=colors["bg_dark"], foreground=colors["text_muted"], anchor="w", padding=(10, 6))
        style.map("Nav.TButton", background=[("active", colors["bg_panel"]), ("selected", colors["bg_panel"])],
                   foreground=[("active", colors["text_main"]), ("selected", colors["text_main"])])

        style.configure("Treeview", background=colors["bg_panel"], fieldbackground=colors["bg_panel"], foreground=colors["text_main"], rowheight=22, font=cls.FONTS["body"])
        style.configure("Treeview.Heading", background=colors["bg_card"], foreground=colors["text_muted"], font=cls.FONTS["body_sm"], relief="flat")
        style.map("Treeview.Heading", background=[("active", colors["bg_card"])] )

        style.configure("Vertical.TScrollbar", gripcount=0, background=colors["bg_dark"], troughcolor=colors["bg_panel"], bordercolor=colors["bg_panel"], arrowsize=0)

        return colors

# Legacy compatibility alias
Theme = ModernTheme
