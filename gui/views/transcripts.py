import tkinter as tk
from tkinter import ttk
from gui.views.base import BaseView


class TranscriptsView(BaseView):
    def _build(self):
        header = ttk.Frame(self, style="Main.TFrame")
        header.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(header, text="Transcripts", style="Header.TLabel").pack(side=tk.LEFT)

        filter_frame = ttk.Frame(header, style="Main.TFrame")
        filter_frame.pack(side=tk.RIGHT)

        ttk.Label(filter_frame, text="Filter:", style="Muted.TLabel").pack(side=tk.LEFT, padx=(0, 3))
        self.filter_var = tk.StringVar()
        filter_entry = ttk.Entry(filter_frame, textvariable=self.filter_var, width=24)
        filter_entry.pack(side=tk.LEFT)
        filter_entry.bind("<KeyRelease>", lambda _: self.call('filter_transcripts', self.filter_var.get()))

        toolbar = ttk.Frame(self, style="Main.TFrame")
        toolbar.pack(fill=tk.X, pady=(0, 6))

        actions = [
            ("↻ Refresh", 'fetch_transcripts'),
            ("Sync", 'sync_selected'),
            ("Delete", 'delete_selected'),
            ("View", 'view_transcript'),
            ("Details", 'view_details'),
            ("Export", 'export_selected'),
        ]
        for text, action in actions:
            ttk.Button(toolbar, text=text, command=lambda a=action: self.call(a)).pack(side=tk.LEFT, padx=(0, 3))

        columns = ("name", "date", "time", "duration", "id")
        self.tree = ttk.Treeview(self, columns=columns, show="headings", style="Treeview")
        for col in columns:
            self.tree.heading(col, text=col.title())
        self.tree.column("name", width=280)
        self.tree.column("date", width=80)
        self.tree.column("time", width=60)
        self.tree.column("duration", width=60)
        self.tree.column("id", width=80)
        self.tree.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.tree.yview, style="Vertical.TScrollbar")
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.place(relx=1.0, rely=0, relheight=1.0, anchor='ne')

    def populate(self, recordings):
        self.tree.delete(*self.tree.get_children())
        for rec in recordings:
            self.tree.insert('', tk.END, iid=rec.get('id'), values=(
                rec.get('display_name'),
                rec.get('display_date'),
                rec.get('display_time'),
                rec.get('display_duration'),
                rec.get('short_id'),
            ))
