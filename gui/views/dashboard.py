from tkinter import ttk
from gui.views.base import BaseView
from gui.components.stat_card import StatCard


class DashboardView(BaseView):
    def _build(self):
        stats_frame = ttk.Frame(self, style="Main.TFrame")
        stats_frame.pack(fill="x")
        stats_frame.columnconfigure((0, 1, 2, 3), weight=1, uniform="stat")

        self.cards = {
            'auth': StatCard(stats_frame, "Auth Status", "🔐", "Checking..."),
            'recordings': StatCard(stats_frame, "Recordings", "🎧", 0),
            'pinecone': StatCard(stats_frame, "Pinecone Vectors", "🌲", 0),
            'last_sync': StatCard(stats_frame, "Last Sync", "⏱", "—"),
        }

        for idx, card in enumerate(self.cards.values()):
            card.grid(row=0, column=idx, padx=4, pady=4, sticky="nsew")

        quick_actions = ttk.Frame(self, style="Main.TFrame")
        quick_actions.pack(fill="x", pady=(10, 0))
        ttk.Label(quick_actions, text="Quick Actions", style="Header.TLabel").pack(anchor="w", pady=(0, 4))

        buttons = [
            ("🔄 Sync All", 'sync_all'),
            ("🧠 Mind Map", 'generate_mindmap'),
            ("🔍 Semantic Search", 'goto_search'),
            ("⚙️ Settings", 'goto_settings'),
        ]
        btn_row = ttk.Frame(quick_actions, style="Main.TFrame")
        btn_row.pack(fill="x")
        for text, action in buttons:
            ttk.Button(btn_row, text=text, style="Accent.TButton", command=lambda a=action: self.call(a)).pack(side="left", padx=3)

    def update_stats(self, stats: dict):
        mapping = {
            'auth': ('auth', lambda v: 'Authenticated' if v else 'Not Auth'),
            'recordings': ('recordings', str),
            'pinecone': ('pinecone', str),
            'last_sync': ('last_sync', str),
        }
        for key, (card_key, transform) in mapping.items():
            if key in stats:
                self.cards[card_key].update_value(transform(stats[key]))
