import tkinter as tk
from tkinter import ttk
from gui.views.base import BaseView
from gui.components.stat_card import StatCard


class DashboardView(BaseView):
    def _build(self):
        hero = ttk.Frame(self, style="Main.TFrame")
        hero.pack(fill="x", pady=(0, 8))
        ttk.Label(hero, text="Dashboard", style="Header.TLabel").pack(anchor="w")
        ttk.Label(hero, text="At-a-glance health of your Plaud workspace", style="Muted.TLabel").pack(anchor="w")

        # Lightweight health strip for quick reading
        strip = ttk.Frame(self, style="Main.TFrame")
        strip.pack(fill="x", pady=(0, 10))
        self.health_badges = {
            'auth': ttk.Label(strip, text="Auth • Checking", style="Badge.TLabel"),
            'pipeline': ttk.Label(strip, text="Pipeline • —", style="Badge.TLabel"),
            'pinecone': ttk.Label(strip, text="Pinecone • —", style="Badge.TLabel"),
            'env': ttk.Label(strip, text="Env • —", style="Badge.TLabel"),
        }
        for badge in self.health_badges.values():
            badge.pack(side="left", padx=(0, 6))

        stats_frame = ttk.Frame(self, style="Main.TFrame")
        stats_frame.pack(fill="x")
        stats_frame.columnconfigure((0, 1, 2, 3, 4), weight=1, uniform="stat")

        self.cards = {
            'auth': StatCard(stats_frame, "Auth Status", "🔐", "Checking..."),
            'recordings': StatCard(stats_frame, "Recordings", "🎧", 0),
            'pinecone': StatCard(stats_frame, "Pinecone Vectors", "🌲", 0),
            'namespaces': StatCard(stats_frame, "Namespaces", "🗂", 0),
            'entities': StatCard(stats_frame, "Graph Entities", "🕸️", 0),
            'last_sync': StatCard(stats_frame, "Last Sync", "⏱", "—"),
        }

        for idx, card in enumerate(self.cards.values()):
            card.grid(row=0, column=idx, padx=4, pady=4, sticky="nsew")

        # Pinecone snapshot
        pinecone_box = ttk.LabelFrame(self, text="Pinecone Snapshot", padding=8, style="Panel.TLabelframe")
        pinecone_box.pack(fill="x", pady=(10, 4))
        pinecone_box.columnconfigure((0, 1), weight=1)

        self.pinecone_labels = {
            'vectors': ttk.Label(pinecone_box, text="Vectors: —", style="Muted.TLabel"),
            'namespaces': ttk.Label(pinecone_box, text="Namespaces: —", style="Muted.TLabel"),
            'dimension': ttk.Label(pinecone_box, text="Dim: —", style="Muted.TLabel"),
            'metric': ttk.Label(pinecone_box, text="Metric: —", style="Muted.TLabel"),
            'mismatch': ttk.Label(pinecone_box, text="", style="Muted.TLabel", foreground="#e67e22"),
            'index': ttk.Label(pinecone_box, text="Index: —", style="Muted.TLabel"),
            'provider': ttk.Label(pinecone_box, text="Provider: —", style="Muted.TLabel"),
            'namespace': ttk.Label(pinecone_box, text="Namespace: —", style="Muted.TLabel"),
        }
        self.pinecone_labels['vectors'].grid(row=0, column=0, sticky="w", pady=2)
        self.pinecone_labels['namespaces'].grid(row=0, column=1, sticky="w", pady=2)
        self.pinecone_labels['dimension'].grid(row=1, column=0, sticky="w", pady=2)
        self.pinecone_labels['metric'].grid(row=1, column=1, sticky="w", pady=2)
        self.pinecone_labels['index'].grid(row=2, column=0, sticky="w", pady=2)
        self.pinecone_labels['provider'].grid(row=2, column=1, sticky="w", pady=2)
        self.pinecone_labels['namespace'].grid(row=3, column=0, sticky="w", pady=2)
        self.pinecone_labels['mismatch'].grid(row=4, column=0, columnspan=2, sticky="w", pady=2)

        # Index / namespace quick selector (bring control to dashboard)
        selector = ttk.Frame(pinecone_box, style="Main.TFrame")
        selector.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        selector.columnconfigure(1, weight=1)
        selector.columnconfigure(3, weight=1)

        ttk.Label(selector, text="Index", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        self.index_var = tk.StringVar()
        self.index_combo = ttk.Combobox(selector, textvariable=self.index_var, state="readonly", width=24)
        self.index_combo.grid(row=0, column=1, sticky="ew", padx=(4, 10))
        self.index_combo.bind("<<ComboboxSelected>>", lambda _: self._on_index_change())

        ttk.Label(selector, text="Namespace", style="Muted.TLabel").grid(row=0, column=2, sticky="w")
        self.namespace_var = tk.StringVar()
        self.namespace_combo = ttk.Combobox(selector, textvariable=self.namespace_var, state="readonly", width=18)
        self.namespace_combo.grid(row=0, column=3, sticky="ew", padx=(4, 0))
        self.namespace_combo.bind("<<ComboboxSelected>>", lambda _: self._on_namespace_change())

        self.refresh_btn = ttk.Button(selector, text="↻ Reload", command=lambda: self.call('load_pinecone_indexes'))
        self.refresh_btn.grid(row=0, column=4, padx=(8, 0))

        # Pipeline status breakdown (raw -> processed -> indexed)
        pipeline = ttk.LabelFrame(self, text="Pipeline Status", padding=8, style="Panel.TLabelframe")
        pipeline.pack(fill="x", pady=(4, 4))
        pipeline.columnconfigure((0, 1, 2), weight=1)
        self.pipeline_labels = {
            'raw': ttk.Label(pipeline, text="Raw: —", style="Muted.TLabel"),
            'processed': ttk.Label(pipeline, text="Processed: —", style="Muted.TLabel"),
            'indexed': ttk.Label(pipeline, text="Indexed: —", style="Muted.TLabel"),
        }
        self.pipeline_labels['raw'].grid(row=0, column=0, sticky="w", pady=2)
        self.pipeline_labels['processed'].grid(row=0, column=1, sticky="w", pady=2)
        self.pipeline_labels['indexed'].grid(row=0, column=2, sticky="w", pady=2)

        # Visual progress indicator for pipeline completeness
        self.pipeline_progress = ttk.Progressbar(pipeline, orient=tk.HORIZONTAL, mode="determinate", maximum=100, value=0)
        self.pipeline_progress.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        self.pipeline_progress_label = ttk.Label(pipeline, text="Waiting for data", style="Muted.TLabel")
        self.pipeline_progress_label.grid(row=2, column=0, columnspan=3, sticky="w", pady=(2, 0))

        # Notion Integration Panel
        notion_panel = ttk.LabelFrame(self, text="📓 Notion Integration", padding=8, style="Panel.TLabelframe")
        notion_panel.pack(fill="x", pady=(4, 4))
        notion_panel.columnconfigure((0, 1), weight=1)
        
        # Status labels
        self.notion_labels = {
            'status': ttk.Label(notion_panel, text="Status: Checking...", style="Muted.TLabel"),
            'database': ttk.Label(notion_panel, text="Database: —", style="Muted.TLabel"),
            'synced': ttk.Label(notion_panel, text="Pages Synced: —", style="Muted.TLabel"),
            'last_sync': ttk.Label(notion_panel, text="Last Sync: Never", style="Muted.TLabel"),
        }
        self.notion_labels['status'].grid(row=0, column=0, sticky="w", pady=2)
        self.notion_labels['database'].grid(row=0, column=1, sticky="w", pady=2)
        self.notion_labels['synced'].grid(row=1, column=0, sticky="w", pady=2)
        self.notion_labels['last_sync'].grid(row=1, column=1, sticky="w", pady=2)
        
        # Action buttons row
        notion_btns = ttk.Frame(notion_panel, style="Panel.TFrame")
        notion_btns.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        
        ttk.Button(notion_btns, text="📤 Push to Notion", 
                   command=lambda: self.call('notion_push')).pack(side="left", padx=2)
        ttk.Button(notion_btns, text="📥 Pull from Notion", 
                   command=lambda: self.call('notion_pull')).pack(side="left", padx=2)
        ttk.Button(notion_btns, text="🔄 Full Sync", 
                   command=lambda: self.call('notion_full_sync')).pack(side="left", padx=2)
        ttk.Button(notion_btns, text="🔍 Check Status", 
                   command=lambda: self.call('notion_check_status')).pack(side="left", padx=2)
        ttk.Button(notion_btns, text="⚙️ Configure", 
                   command=lambda: self.call('notion_configure')).pack(side="left", padx=2)

        # Recent recordings
        recent = ttk.LabelFrame(self, text="Recent Recordings", padding=8, style="Panel.TLabelframe")
        recent.pack(fill="both", expand=True, pady=(4, 8))
        recent.columnconfigure(0, weight=1)
        recent.rowconfigure(1, weight=1)

        ttk.Label(recent, text="Latest sync pulls", style="Muted.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))

        cols = ("name", "date", "duration")
        self.recent_tree = ttk.Treeview(recent, columns=cols, show="headings", height=6, style="Treeview")
        self.recent_tree.heading("name", text="Name")
        self.recent_tree.heading("date", text="Date")
        self.recent_tree.heading("duration", text="Duration")
        self.recent_tree.column("name", width=320, anchor="w")
        self.recent_tree.column("date", width=100, anchor="center")
        self.recent_tree.column("duration", width=90, anchor="center")
        self.recent_tree.grid(row=1, column=0, sticky="nsew")

        # Activity feed (shows latest log lines)
        activity = ttk.LabelFrame(self, text="Activity", padding=8, style="Panel.TLabelframe")
        activity.pack(fill="both", expand=True, pady=(0, 8))
        activity.columnconfigure(0, weight=1)
        ttk.Label(activity, text="Latest operations and status messages", style="Muted.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 4))
        self.activity_list = tk.Text(activity, height=6, wrap=tk.NONE, state="disabled", font=("JetBrains Mono", 9))
        self.activity_list.grid(row=1, column=0, sticky="nsew")
        activity.rowconfigure(1, weight=1)
        # scrollbar
        scroll = ttk.Scrollbar(activity, orient=tk.VERTICAL, command=self.activity_list.yview)
        scroll.grid(row=1, column=1, sticky="ns")
        self.activity_list.configure(yscrollcommand=scroll.set)

        quick_actions = ttk.LabelFrame(self, text="Quick Actions", padding=8, style="Panel.TLabelframe")
        quick_actions.pack(fill="x", pady=(12, 0))
        ttk.Label(quick_actions, text="Common flows in one click", style="Muted.TLabel").pack(anchor="w", pady=(0, 6))

        buttons = [
            ("🔄 Sync All", 'sync_all'),
            ("📤 Notion Sync", 'sync_to_notion'),
            ("🕸️ Knowledge Graph", 'goto_knowledge_graph'),
            ("🔍 Semantic Search", 'goto_search'),
            ("↻ Refresh", 'refresh_dashboard'),
        ]
        btn_row = ttk.Frame(quick_actions, style="Panel.TFrame")
        btn_row.pack(fill="x")
        for idx, (text, action) in enumerate(buttons):
            style = "Accent.TButton" if idx == 0 else "TButton"
            ttk.Button(btn_row, text=text, style=style, command=lambda a=action: self.call(a)).pack(side="left", padx=3)

        # Inline recommendations / next steps
        self.next_steps = ttk.Label(self, text="", style="Muted.TLabel")
        self.next_steps.pack(anchor="w", pady=(6, 0))

    def update_stats(self, stats: dict):
        mapping = {
            'auth': ('auth', lambda v: 'Authenticated' if v else 'Not Auth'),
            'recordings': ('recordings', str),
            'pinecone': ('pinecone', str),
            'pinecone_namespaces': ('namespaces', str),
            'graph_entities': ('entities', lambda v: f"{v} entities"),
            'last_sync': ('last_sync', str),
        }
        for key, (card_key, transform) in mapping.items():
            if key in stats:
                self.cards[card_key].update_value(transform(stats[key]))

        # Health strip badges
        if 'auth' in stats:
            badge = self.health_badges['auth']
            if stats.get('auth'):
                badge.configure(text="Auth • Connected", style="SuccessBadge.TLabel")
            else:
                badge.configure(text="Auth • Not signed in", style="WarnBadge.TLabel")
        if any(k in stats for k in ('status_raw', 'status_processed', 'status_indexed')):
            raw = stats.get('status_raw', 0) or 0
            proc = stats.get('status_processed', 0) or 0
            idx = stats.get('status_indexed', 0) or 0
            total = raw + proc + idx
            badge = self.health_badges['pipeline']
            badge.configure(text=f"Pipeline • {total} items", style="Badge.TLabel")

        if any(k in stats for k in ('pinecone', 'pinecone_namespaces', 'pinecone_dim_mismatch')):
            mismatch = stats.get('pinecone_dim_mismatch')
            pine_badge = self.health_badges['pinecone']
            if mismatch:
                pine_badge.configure(text="Pinecone • Needs attention", style="WarnBadge.TLabel")
            else:
                pine_badge.configure(text="Pinecone • Healthy", style="SuccessBadge.TLabel")

        if any(k in stats for k in ('pinecone', 'pinecone_namespaces', 'pinecone_dim', 'pinecone_metric', 'pinecone_dim_mismatch', 'pinecone_index', 'pinecone_provider')):
            self.update_pinecone_snapshot({
                'vectors': stats.get('pinecone'),
                'namespaces': stats.get('pinecone_namespaces'),
                'dimension': stats.get('pinecone_dim'),
                'metric': stats.get('pinecone_metric'),
                'dim_mismatch': stats.get('pinecone_dim_mismatch'),
                'index': stats.get('pinecone_index'),
                'provider': stats.get('pinecone_provider'),
                'namespace': stats.get('pinecone_namespace'),
            })

        if any(k in stats for k in ('status_raw', 'status_processed', 'status_indexed')):
            self.update_pipeline_breakdown(
                raw=stats.get('status_raw'),
                processed=stats.get('status_processed'),
                indexed=stats.get('status_indexed'),
            )

        # Inline recommendations to nudge the user
        if stats.get('pinecone_dim_mismatch'):
            self.next_steps.configure(text="⚠️ Dimension mismatch detected — open Pinecone tab and click Auto-fix.")
        elif stats.get('recordings', 0) == 0:
            self.next_steps.configure(text="📥 No recordings yet — click Fetch to pull Plaud transcripts.")
        else:
            self.next_steps.configure(text="✅ Everything looks good. Try Semantic Search or Sync All.")

        self._update_env_badge()

    def update_recent_transcripts(self, transcripts):
        if not hasattr(self, 'recent_tree'):
            return
        for item in self.recent_tree.get_children():
            self.recent_tree.delete(item)
        # Sort newest first by date/time if present
        sorted_recs = sorted(transcripts, key=lambda r: (r.get('display_date', ''), r.get('display_time', '')), reverse=True)
        for rec in sorted_recs[:6]:
            self.recent_tree.insert('', 'end', values=(rec.get('display_name', '—'), rec.get('display_date', '—'), rec.get('display_duration', '—')))

    def update_pinecone_snapshot(self, snapshot: dict):
        if not snapshot:
            return
        if snapshot.get('vectors') is not None:
            self.pinecone_labels['vectors'].configure(text=f"Vectors: {snapshot.get('vectors')}")
        if snapshot.get('namespaces') is not None:
            self.pinecone_labels['namespaces'].configure(text=f"Namespaces: {snapshot.get('namespaces')}")
        if snapshot.get('dimension') is not None:
            self.pinecone_labels['dimension'].configure(text=f"Dim: {snapshot.get('dimension')}")
        if snapshot.get('metric') is not None:
            self.pinecone_labels['metric'].configure(text=f"Metric: {snapshot.get('metric')}")
        if snapshot.get('index') is not None:
            self.pinecone_labels['index'].configure(text=f"Index: {snapshot.get('index')}")
        if snapshot.get('provider') is not None:
            self.pinecone_labels['provider'].configure(text=f"Provider: {snapshot.get('provider')}")
        if snapshot.get('namespace') is not None:
            self.pinecone_labels['namespace'].configure(text=f"Namespace: {snapshot.get('namespace')}")
        
        # Color-coded mismatch indicator
        if snapshot.get('dim_mismatch'):
            self.pinecone_labels['mismatch'].configure(
                text="🟡 Dimension mismatch — go to Pinecone tab and click Auto-fix",
                foreground="#e67e22"
            )
        else:
            self.pinecone_labels['mismatch'].configure(
                text="🟢 Index aligned with provider",
                foreground="#27ae60"
            )

        # Sync selectors when snapshot changes
        if snapshot.get('index') and self.index_combo is not None:
            try:
                self.index_var.set(snapshot.get('index'))
            except Exception:
                pass
        if snapshot.get('namespace') and hasattr(self, 'namespace_var'):
            try:
                self.namespace_var.set(snapshot.get('namespace'))
            except Exception:
                pass

    def update_pipeline_breakdown(self, raw=None, processed=None, indexed=None):
        if raw is not None:
            self.pipeline_labels['raw'].configure(text=f"Raw: {raw}")
        if processed is not None:
            self.pipeline_labels['processed'].configure(text=f"Processed: {processed}")
        if indexed is not None:
            self.pipeline_labels['indexed'].configure(text=f"Indexed: {indexed}")

        # Reflect overall completion in a single glance
        counts = [v for v in (raw, processed, indexed) if isinstance(v, int)]
        if counts and len(counts) == 3:
            total = sum(counts)
            if total:
                completion = int((counts[2] / total) * 100)
                self.pipeline_progress.configure(value=completion)
                self.pipeline_progress_label.configure(text=f"{completion}% indexed · {counts[2]} / {total}")
            else:
                self.pipeline_progress.configure(value=0)
                self.pipeline_progress_label.configure(text="Waiting for data")
        else:
            self.pipeline_progress.configure(value=0)
            self.pipeline_progress_label.configure(text="Waiting for data")

    def update_notion_status(self, status: dict):
        """Update Notion integration panel with current status."""
        if not hasattr(self, 'notion_labels'):
            return
        
        if status.get('connected'):
            self.notion_labels['status'].configure(
                text="Status: 🟢 Connected", 
                foreground="#27ae60"
            )
        elif status.get('error'):
            self.notion_labels['status'].configure(
                text=f"Status: 🔴 {status.get('error', 'Error')[:30]}", 
                foreground="#e74c3c"
            )
        else:
            self.notion_labels['status'].configure(
                text="Status: ⚪ Not configured", 
                foreground="#7f8c8d"
            )
        
        if status.get('database_id'):
            db_id = status['database_id']
            self.notion_labels['database'].configure(text=f"Database: {db_id[:8]}...")
        
        if status.get('pages_synced') is not None:
            self.notion_labels['synced'].configure(text=f"Pages Synced: {status['pages_synced']}")
        
        if status.get('last_sync'):
            self.notion_labels['last_sync'].configure(text=f"Last Sync: {status['last_sync']}")

    def update_activity(self, lines):
        if not hasattr(self, 'activity_list'):
            return
        self.activity_list.configure(state="normal")
        self.activity_list.delete('1.0', tk.END)
        self.activity_list.insert('1.0', "\n".join(lines))
        self.activity_list.configure(state="disabled")

    def _update_env_badge(self):
        """Quick environment readiness summary for dashboard strip."""
        try:
            import os
            cid = os.getenv("PLAUD_CLIENT_ID")
            secret = os.getenv("PLAUD_CLIENT_SECRET")
            pkey = os.getenv("PINECONE_API_KEY")
            gem = os.getenv("GEMINI_API_KEY")
            openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")

            missing_core = [name for name, val in {"Plaud ID": cid, "Plaud Secret": secret}.items() if not val]
            if missing_core:
                self.health_badges['env'].configure(text="Env • Missing keys", style="WarnBadge.TLabel")
                return

            if not pkey:
                self.health_badges['env'].configure(text="Env • Pinecone key missing", style="WarnBadge.TLabel")
                return

            if not (gem or openai_key):
                self.health_badges['env'].configure(text="Env • LLM key missing", style="WarnBadge.TLabel")
                return

            self.health_badges['env'].configure(text="Env • Ready", style="SuccessBadge.TLabel")
        except Exception:
            self.health_badges['env'].configure(text="Env • Check", style="WarnBadge.TLabel")

    # ------------------------------------------------------------------
    # Pinecone selectors
    # ------------------------------------------------------------------
    def set_indexes(self, indexes: list, current: str):
        if hasattr(self, 'index_combo'):
            self.index_combo['values'] = indexes
            if current:
                self.index_var.set(current if current in indexes else (indexes[0] if indexes else ''))

    def set_namespaces(self, namespaces: list, current: str = None):
        if hasattr(self, 'namespace_combo'):
            self.namespace_combo['values'] = namespaces
            if namespaces:
                target = current if current in namespaces else namespaces[0]
                self.namespace_var.set(target)

    def _on_index_change(self):
        val = self.index_var.get()
        if val:
            self.call('change_index', val)

    def _on_namespace_change(self):
        val = self.namespace_var.get()
        if val:
            self.call('select_namespace', val)
