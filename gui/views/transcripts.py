import tkinter as tk
from tkinter import ttk
from gui.utils.tooltips import ToolTip
from gui.views.base import BaseView


class TranscriptsView(BaseView):
    def _build(self):
        # Persist the most recent dataset for local filtering
        self.recordings = []
        self.status_filter = "all"
        self.status_buttons = {}

        header = ttk.Frame(self, style="Main.TFrame")
        header.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(header, text="Transcripts", style="Header.TLabel").pack(side=tk.LEFT)
        ttk.Label(header, text="Browse and act on Plaud recordings", style="Muted.TLabel").pack(side=tk.LEFT, padx=(8, 0))

        filter_frame = ttk.Frame(header, style="Main.TFrame")
        filter_frame.pack(side=tk.RIGHT)

        ttk.Label(filter_frame, text="Filter:", style="Muted.TLabel").pack(side=tk.LEFT, padx=(0, 3))
        self.filter_var = tk.StringVar()
        filter_entry = ttk.Entry(filter_frame, textvariable=self.filter_var, width=24)
        filter_entry.pack(side=tk.LEFT)
        filter_entry.bind("<KeyRelease>", lambda _: self.call('filter_transcripts', self.filter_var.get()))
        ttk.Button(filter_frame, text="Clear", command=lambda: (self.filter_var.set(""), self.call('filter_transcripts', "")), style="Pill.TButton").pack(side=tk.LEFT, padx=(6, 0))

        # Quick status chips (All/Raw/Processed/Indexed)
        chips = ttk.Frame(header, style="Main.TFrame")
        chips.pack(side=tk.RIGHT, padx=(12, 0))
        for key, label in [
            ("all", "All"),
            ("raw", "Raw"),
            ("processed", "Processed"),
            ("indexed", "Indexed"),
        ]:
            btn = ttk.Button(chips, text=label, style="Pill.TButton", command=lambda k=key: self._set_status_filter(k))
            btn.pack(side=tk.LEFT, padx=3)
            self.status_buttons[key] = btn

        toolbar = ttk.LabelFrame(self, text="Actions", padding=6, style="Panel.TLabelframe")
        toolbar.pack(fill=tk.X, pady=(0, 8))

        actions = [
            ("↻ Refresh", 'fetch_transcripts'),
            ("Sync", 'sync_selected'),
            ("Sync All", 'sync_all'),
            ("Delete", 'delete_selected'),
            ("View", 'view_transcript'),
            ("Details", 'view_details'),
            ("Export", 'export_selected'),
            ("DB", 'show_db_browser'),
            ("Copy meta", 'copy_metadata'),
        ]
        for text, action in actions:
            btn = ttk.Button(toolbar, text=text, command=lambda a=action: self.call(a))
            btn.pack(side=tk.LEFT, padx=(0, 4), pady=2)
            ToolTip(btn, {
                'fetch_transcripts': "Fetch recordings from Plaud into local DB",
                'sync_selected': "Chunk, embed, and upsert the selected recording",
                'sync_all': "Process all pending recordings",
                'delete_selected': "Remove vectors for the selected recording from Pinecone",
                'view_transcript': "Show metadata for the selected recording",
                'view_details': "Open transcript text",
                'export_selected': "Export transcript + metadata to JSON",
                'show_db_browser': "Inspect the local SQLite database contents",
            }.get(action, text))

        table_frame = ttk.LabelFrame(self, text="Transcript List", padding=6, style="Panel.TLabelframe")
        table_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("name", "status", "source", "date", "time", "duration", "id")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", style="Treeview")
        for col in columns:
            self.tree.heading(col, text=col.title())
        self.tree.column("name", width=280)
        self.tree.column("status", width=90)
        self.tree.column("source", width=80)
        self.tree.column("date", width=80)
        self.tree.column("time", width=60)
        self.tree.column("duration", width=70)
        self.tree.column("id", width=80)
        self.tree.pack(fill=tk.BOTH, expand=True)

        # Color-coded row tags for status readability
        self.tree.tag_configure("raw", foreground="#facc15")
        self.tree.tag_configure("processed", foreground="#38bdf8")
        self.tree.tag_configure("indexed", foreground="#34d399")

        # Double-click to open details quickly
        self.tree.bind("<Double-1>", lambda _: self.call('view_details'))
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview, style="Vertical.TScrollbar")
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Empty state label overlays the table when no data
        self.empty_label = ttk.Label(table_frame, text="No transcripts yet", style="Muted.TLabel")
        self.empty_label.place(relx=0.5, rely=0.5, anchor="center")
        self._toggle_empty(True)

        # Details side-car to keep context visible
        detail = ttk.LabelFrame(self, text="Selection", padding=8, style="Panel.TLabelframe")
        detail.pack(fill=tk.X, pady=(6, 0))
        detail.columnconfigure(1, weight=1)
        self.detail_labels = {}
        for row, (label, key) in enumerate([
            ("Title", "display_name"),
            ("Status", "status"),
            ("Date", "display_date"),
            ("Time", "display_time"),
            ("Duration", "display_duration"),
            ("ID", "short_id"),
            ("Source", "source"),
            ("Summary", "plaud_summary"),
            ("Outline", "plaud_outline"),
            ("Keywords", "plaud_keywords"),
        ]):
            ttk.Label(detail, text=f"{label}:", style="Muted.TLabel").grid(row=row, column=0, sticky="w", pady=2, padx=(0, 8))
            val = ttk.Label(detail, text="—", style="TLabel")
            val.grid(row=row, column=1, sticky="w", pady=2)
            self.detail_labels[key] = val

        self._set_status_filter("all")

    def populate(self, recordings):
        # Store dataset locally for fast status toggling
        self.recordings = recordings or []
        self._apply_filters()
        self._update_status_chips()
        
        # Update empty state message based on whether we have any data
        if len(self.recordings) == 0:
            self.empty_label.config(text="No transcripts loaded.\nCheck Plaud API status or wait for server recovery.")
        else:
            self.empty_label.config(text="No transcripts match current filters")
        
        self._toggle_empty(len(self.recordings) == 0)
        self._on_select()  # refresh detail panel

    def get_selected_ids(self):
        return list(self.tree.selection())

    def _toggle_empty(self, show: bool):
        if show:
            self.empty_label.lift()
            self.empty_label.place(relx=0.5, rely=0.5, anchor="center")
        else:
            self.empty_label.place_forget()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _set_status_filter(self, status: str):
        """Switch the active status chip and refresh the table."""
        self.status_filter = status
        for key, btn in self.status_buttons.items():
            style = "PillActive.TButton" if key == status else "Pill.TButton"
            btn.configure(style=style)
        self._apply_filters()

    def _apply_filters(self):
        """Render rows matching the current status filter."""
        self.tree.delete(*self.tree.get_children())
        for rec in self.recordings:
            status = rec.get('status', 'raw')
            if self.status_filter != "all" and status != self.status_filter:
                continue
            self.tree.insert('', tk.END, iid=rec.get('id'), values=(
                rec.get('display_name'),
                status,
                rec.get('source', '—'),
                rec.get('display_date'),
                rec.get('display_time'),
                rec.get('display_duration'),
                rec.get('short_id'),
            ), tags=(status,))
        # Keep global filtered state in sync for downstream actions
        filtered_records = [rec for rec in self.recordings
                    if self.status_filter == "all" or rec.get('status', 'raw') == self.status_filter]
        self.state.filtered_transcripts = filtered_records
        self._toggle_empty(len(self.tree.get_children()) == 0)

    def _update_status_chips(self):
        """Show counts inside each status chip for quick scanning."""
        counts = {"all": len(self.recordings), "raw": 0, "processed": 0, "indexed": 0}
        for rec in self.recordings:
            key = rec.get('status', 'raw')
            counts[key] = counts.get(key, 0) + 1
        for key, btn in self.status_buttons.items():
            base = btn.cget("text").split("(")[0].strip()
            btn.configure(text=f"{base} ({counts.get(key, 0)})")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def copy_metadata(self):
        """Copy selected recording metadata and Plaud extras to clipboard."""
        selection = self.get_selected_ids()
        if not selection:
            return
        rec_id = selection[0]
        found = next((r for r in self.recordings if r.get('id') == rec_id), None)
        if not found:
            return
        # Flatten interesting fields
        fields = {
            'id': found.get('id'),
            'display_name': found.get('display_name'),
            'status': found.get('status'),
            'date': found.get('display_date'),
            'time': found.get('display_time'),
            'duration': found.get('display_duration'),
            'source': found.get('source'),
            'plaud_summary': found.get('plaud_summary'),
            'plaud_outline': found.get('plaud_outline'),
            'plaud_keywords': found.get('plaud_keywords'),
        }
        import json
        payload = json.dumps(fields, indent=2, ensure_ascii=False)
        try:
            self.clipboard_clear()
            self.clipboard_append(payload)
        except Exception:
            pass

    def _on_select(self, event=None):
        """Update the detail card with the selected row."""
        selection = self.get_selected_ids()
        if not selection:
            for lbl in self.detail_labels.values():
                lbl.configure(text="—")
            return
        rec_id = selection[0]
        found = next((r for r in self.recordings if r.get('id') == rec_id), None)
        if not found:
            return
        mapping = {
            'display_name': '—',
            'status': '—',
            'display_date': '—',
            'display_time': '—',
            'display_duration': '—',
            'short_id': '—',
            'source': '—',
            'plaud_summary': '—',
            'plaud_outline': '—',
            'plaud_keywords': '—',
        }
        mapping.update({k: found.get(k, v) for k, v in mapping.items()})
        for key, lbl in self.detail_labels.items():
            lbl.configure(text=mapping.get(key, '—'))
