"""
Enhanced Pinecone Vector Workspace with full SDK granularity.

Features:
- Index/namespace selection with stats
- Metadata filtering (JSON syntax)
- Similarity search with top_k control
- Vector preview, edit, delete
- Pagination for large datasets
- Export capabilities
"""
import json
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from gui.views.base import BaseView
from gui.theme import ModernTheme


class PineconeView(BaseView):
    def _build(self):
        self.current_page = 1
        self.page_size = 100
        self.all_vectors = []  # full cached list for client-side filtering

        # ─────────────────────────────────────────────────────────────
        # TOP: Index selector + stats cards
        # ─────────────────────────────────────────────────────────────
        top = ttk.Frame(self, style="Main.TFrame")
        top.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(top, text="Vector Workspace", style="Header.TLabel").pack(side=tk.LEFT)

        controls = ttk.Frame(top, style="Main.TFrame")
        controls.pack(side=tk.RIGHT)

        # Index dropdown
        ttk.Label(controls, text="Index", style="Muted.TLabel").pack(side=tk.LEFT, padx=(0, 2))
        self.index_var = tk.StringVar()
        self.index_dropdown = ttk.Combobox(controls, textvariable=self.index_var, state="readonly", width=16)
        self.index_dropdown.pack(side=tk.LEFT)
        self.index_dropdown.bind("<<ComboboxSelected>>", lambda _: self.call("change_index", self.index_var.get()))

        # Namespace dropdown
        ttk.Label(controls, text="NS", style="Muted.TLabel").pack(side=tk.LEFT, padx=(8, 2))
        self.namespace_var = tk.StringVar()
        self.namespace_dropdown = ttk.Combobox(controls, textvariable=self.namespace_var, state="readonly", width=12)
        self.namespace_dropdown.pack(side=tk.LEFT)
        self.namespace_dropdown.bind("<<ComboboxSelected>>", lambda _: self.call("select_namespace", self.namespace_var.get()))

        # Stats row
        stats_row = ttk.Frame(self, style="Main.TFrame")
        stats_row.pack(fill=tk.X, pady=(0, 4))

        self.stat_labels = {}
        for key, label in [("vectors", "Vec"), ("dimension", "Dim"), ("metric", "Metric"), ("namespaces", "NS")]:
            card = ttk.Frame(stats_row, style="Card.TFrame", padding=4)
            card.pack(side=tk.LEFT, padx=(0, 4))
            ttk.Label(card, text=label, style="CardTitle.TLabel").pack(anchor="w")
            lbl = ttk.Label(card, text="—", style="TLabel")
            lbl.pack(anchor="w")
            self.stat_labels[key] = lbl

        # ─────────────────────────────────────────────────────────────
        # ACTION BUTTONS
        # ─────────────────────────────────────────────────────────────
        actions = ttk.Frame(self, style="Main.TFrame")
        actions.pack(fill=tk.X, pady=(0, 4))

        for text, cmd in [
            ("↻ Refresh", "refresh_vectors"),
            ("Filter", "open_metadata_filter"),
            ("Search", "open_similarity_search"),
            ("Fetch ID", "fetch_vector_by_id"),
            ("Edit", "edit_vector_metadata"),
            ("Delete", "delete_selected_vectors"),
            ("Export", "export_vectors"),
        ]:
            ttk.Button(actions, text=text, command=lambda c=cmd: self.call(c)).pack(side=tk.LEFT, padx=(0, 3))

        # ─────────────────────────────────────────────────────────────
        # LOCAL FILTER BAR
        # ─────────────────────────────────────────────────────────────
        filter_bar = ttk.Frame(self, style="Main.TFrame")
        filter_bar.pack(fill=tk.X, pady=(0, 3))

        ttk.Label(filter_bar, text="Filter:", style="Muted.TLabel").pack(side=tk.LEFT, padx=(0, 2))
        self.filter_var = tk.StringVar()
        self.filter_entry = ttk.Entry(filter_bar, textvariable=self.filter_var, width=30)
        self.filter_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.filter_entry.bind("<KeyRelease>", lambda _: self._apply_local_filter())

        ttk.Label(filter_bar, text="Page:", style="Muted.TLabel").pack(side=tk.LEFT, padx=(10, 2))
        self.page_size_var = tk.StringVar(value="100")
        page_combo = ttk.Combobox(filter_bar, textvariable=self.page_size_var, values=["50", "100", "500", "1000"], width=5, state="readonly")
        page_combo.pack(side=tk.LEFT)
        page_combo.bind("<<ComboboxSelected>>", lambda _: self._change_page_size())

        # ─────────────────────────────────────────────────────────────
        # MAIN: Treeview + Preview pane
        # ─────────────────────────────────────────────────────────────
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        # Tree
        tree_frame = ttk.Frame(main_pane, style="Main.TFrame")
        columns = ("id", "title", "date", "duration", "themes", "fields")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", selectmode="extended")
        for col, width in [("id", 100), ("title", 200), ("date", 80), ("duration", 60), ("themes", 150), ("fields", 50)]:
            self.tree.heading(col, text=col.upper())
            self.tree.column(col, width=width)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        vsb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        main_pane.add(tree_frame, weight=3)

        # Preview
        preview_frame = ttk.LabelFrame(main_pane, text="Details", padding=4)
        self.preview = scrolledtext.ScrolledText(preview_frame, wrap=tk.WORD, font=("JetBrains Mono", 9), bg="#0f172a", fg="#f8fafc", height=12)
        self.preview.pack(fill=tk.BOTH, expand=True)
        main_pane.add(preview_frame, weight=1)

        # ─────────────────────────────────────────────────────────────
        # PAGINATION
        # ─────────────────────────────────────────────────────────────
        page_bar = ttk.Frame(self, style="Main.TFrame")
        page_bar.pack(fill=tk.X, pady=(4, 0))

        ttk.Button(page_bar, text="◀", command=self._prev_page, width=3).pack(side=tk.LEFT)
        self.page_label = ttk.Label(page_bar, text="1/1", style="Muted.TLabel")
        self.page_label.pack(side=tk.LEFT, padx=6)
        ttk.Button(page_bar, text="▶", command=self._next_page, width=3).pack(side=tk.LEFT)

        self.total_label = ttk.Label(page_bar, text="", style="Muted.TLabel")
        self.total_label.pack(side=tk.RIGHT)

    # ─────────────────────────────────────────────────────────────────
    # Public interface for app.py
    # ─────────────────────────────────────────────────────────────────
    def on_show(self):
        if not self.index_dropdown["values"]:
            self.call("load_pinecone_indexes")

    def set_indexes(self, indexes: list, current: str):
        self.index_dropdown["values"] = indexes
        if current in indexes:
            self.index_var.set(current)
        elif indexes:
            self.index_var.set(indexes[0])

    def set_namespaces(self, namespaces: list):
        self.namespace_dropdown["values"] = namespaces
        if namespaces:
            self.namespace_var.set(namespaces[0])

    def set_stats(self, stats: dict):
        for key in ("vectors", "dimension", "metric", "namespaces"):
            val = stats.get(key, "—")
            if key == "vectors" and isinstance(val, int):
                val = f"{val:,}"
            self.stat_labels[key].configure(text=str(val))

    def populate(self, vectors: list):
        """Cache vectors and display first page."""
        self.all_vectors = vectors
        self.current_page = 1
        self._render_page()
        self.total_label.configure(text=f"Total: {len(vectors):,} vectors")

    # ─────────────────────────────────────────────────────────────────
    # Pagination & filtering
    # ─────────────────────────────────────────────────────────────────
    def _render_page(self):
        self.tree.delete(*self.tree.get_children())
        query = self.filter_var.get().lower()
        filtered = [v for v in self.all_vectors if query in json.dumps(v).lower()] if query else self.all_vectors

        start = (self.current_page - 1) * self.page_size
        end = start + self.page_size
        page_data = filtered[start:end]

        for vec in page_data:
            self.tree.insert("", tk.END, iid=vec["id"], values=(
                vec.get("short_id", vec["id"][:12] + "…"),
                vec.get("title", "Untitled"),
                vec.get("date", "—"),
                vec.get("duration", "—"),
                vec.get("tags", "—"),
                vec.get("field_count", 0),
            ))

        total_pages = max(1, (len(filtered) + self.page_size - 1) // self.page_size)
        self.page_label.configure(text=f"{self.current_page}/{total_pages}")

    def _prev_page(self):
        if self.current_page > 1:
            self.current_page -= 1
            self._render_page()

    def _next_page(self):
        query = self.filter_var.get().lower()
        filtered = [v for v in self.all_vectors if query in json.dumps(v).lower()] if query else self.all_vectors
        total_pages = max(1, (len(filtered) + self.page_size - 1) // self.page_size)
        if self.current_page < total_pages:
            self.current_page += 1
            self._render_page()

    def _apply_local_filter(self):
        self.current_page = 1
        self._render_page()

    def _change_page_size(self):
        self.page_size = int(self.page_size_var.get())
        self.current_page = 1
        self._render_page()

    def _on_select(self, _event):
        selected = self.tree.selection()
        if not selected:
            return
        vec_id = selected[0]
        vec = next((v for v in self.all_vectors if v["id"] == vec_id), None)
        if vec:
            self.preview.configure(state=tk.NORMAL)
            self.preview.delete("1.0", tk.END)
            self.preview.insert("1.0", json.dumps(vec.get("metadata", vec), indent=2))
            self.preview.configure(state=tk.DISABLED)

    # ─────────────────────────────────────────────────────────────────
    # Dialog launchers (delegated to app)
    # ─────────────────────────────────────────────────────────────────
    def get_selected_ids(self) -> list:
        return list(self.tree.selection())
