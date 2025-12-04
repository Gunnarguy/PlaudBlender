"""
Pinecone Vector Workspace - Full SDK integration.

Exposes complete Pinecone SDK functionality:
- Index management (list, describe, stats)
- Namespace management (list, create, delete)
- Vector operations (query, upsert, update, fetch, delete)
- Metadata filtering (fetch_by_metadata)
- Cross-namespace queries (query_namespaces)
- Bulk operations (import, export)
"""
import json
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog, simpledialog
from gui.views.base import BaseView
from gui.theme import ModernTheme


class PineconeView(BaseView):
    def _build(self):
        self.current_page = 1
        self.page_size = 100
        self.all_vectors = []
        # Sorting state
        self.sort_column = "date"
        self.sort_reverse = True  # newest first by default

        # ─────────────────────────────────────────────────────────────
        # ROW 1: Index/Namespace selectors + Stats
        # ─────────────────────────────────────────────────────────────
        row1 = ttk.Frame(self, style="Main.TFrame")
        row1.pack(fill=tk.X, pady=(0, 4))

        # Left: selectors
        sel = ttk.Frame(row1, style="Main.TFrame")
        sel.pack(side=tk.LEFT)

        ttk.Label(sel, text="Index:", style="Muted.TLabel").pack(side=tk.LEFT)
        self.index_var = tk.StringVar()
        self.index_dropdown = ttk.Combobox(sel, textvariable=self.index_var, state="readonly", width=14)
        self.index_dropdown.pack(side=tk.LEFT, padx=(2, 8))
        self.index_dropdown.bind("<<ComboboxSelected>>", lambda _: self.call("change_index", self.index_var.get()))

        ttk.Label(sel, text="NS:", style="Muted.TLabel").pack(side=tk.LEFT)
        self.namespace_var = tk.StringVar()
        self.namespace_dropdown = ttk.Combobox(sel, textvariable=self.namespace_var, state="readonly", width=12)
        self.namespace_dropdown.pack(side=tk.LEFT, padx=(2, 0))
        self.namespace_dropdown.bind("<<ComboboxSelected>>", lambda _: self.call("select_namespace", self.namespace_var.get()))

        # Right: stats
        stats = ttk.Frame(row1, style="Main.TFrame")
        stats.pack(side=tk.RIGHT)
        self.stat_labels = {}
        for key, lbl in [("vectors", "Vec"), ("dimension", "Dim"), ("metric", "Mtrc")]:
            ttk.Label(stats, text=f"{lbl}:", style="Muted.TLabel").pack(side=tk.LEFT, padx=(6, 0))
            val = ttk.Label(stats, text="—", style="TLabel")
            val.pack(side=tk.LEFT, padx=(1, 0))
            self.stat_labels[key] = val

        # ─────────────────────────────────────────────────────────────
        # ROW 2: Primary actions (Query/Search)
        # ─────────────────────────────────────────────────────────────
        row2 = ttk.Frame(self, style="Main.TFrame")
        row2.pack(fill=tk.X, pady=(0, 2))

        ttk.Button(row2, text="↻ Refresh", command=lambda: self.call("refresh_vectors")).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(row2, text="Similarity", command=self._similarity_search_dialog).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(row2, text="By Metadata", command=self._metadata_filter_dialog).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(row2, text="All NS", command=self._query_all_namespaces_dialog).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(row2, text="Fetch ID", command=self._fetch_by_id_dialog).pack(side=tk.LEFT, padx=(0, 2))

        # Separator
        ttk.Separator(row2, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)

        ttk.Button(row2, text="+ Upsert", command=self._upsert_dialog).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(row2, text="Edit Meta", command=self._edit_metadata_dialog).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(row2, text="Delete", command=self._delete_dialog).pack(side=tk.LEFT, padx=(0, 2))

        ttk.Separator(row2, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)

        ttk.Button(row2, text="Bulk Import", command=self._bulk_import_dialog).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(row2, text="Export", command=self._export_dialog).pack(side=tk.LEFT, padx=(0, 2))

        # ─────────────────────────────────────────────────────────────
        # ROW 3: Namespace management
        # ─────────────────────────────────────────────────────────────
        row3 = ttk.Frame(self, style="Main.TFrame")
        row3.pack(fill=tk.X, pady=(0, 4))

        ttk.Label(row3, text="NS Mgmt:", style="Muted.TLabel").pack(side=tk.LEFT)
        ttk.Button(row3, text="+ Create", command=self._create_namespace_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(row3, text="- Delete", command=self._delete_namespace_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(row3, text="List All", command=self._list_namespaces_dialog).pack(side=tk.LEFT, padx=2)

        ttk.Separator(row3, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Label(row3, text="Filter:", style="Muted.TLabel").pack(side=tk.LEFT)
        self.filter_var = tk.StringVar()
        self.filter_entry = ttk.Entry(row3, textvariable=self.filter_var, width=25)
        self.filter_entry.pack(side=tk.LEFT, padx=2)
        self.filter_entry.bind("<KeyRelease>", lambda _: self._apply_local_filter())

        ttk.Label(row3, text="Page:", style="Muted.TLabel").pack(side=tk.LEFT, padx=(8, 0))
        self.page_size_var = tk.StringVar(value="100")
        ttk.Combobox(row3, textvariable=self.page_size_var, values=["50", "100", "500", "1000"], width=5, state="readonly").pack(side=tk.LEFT, padx=2)

        # ─────────────────────────────────────────────────────────────
        # MAIN: Treeview + Preview pane
        # ─────────────────────────────────────────────────────────────
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        tree_frame = ttk.Frame(main_pane, style="Main.TFrame")
        columns = ("id", "title", "date", "dur", "tags", "flds")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", selectmode="extended")
        # Column config with sort keys
        col_config = [
            ("id", 90, "ID", "id"),
            ("title", 180, "Title", "title"),
            ("date", 70, "Date ▼", "date"),  # default sort
            ("dur", 50, "Dur", "duration"),
            ("tags", 140, "Tags", "tags"),
            ("flds", 40, "#", "field_count"),
        ]
        for col, w, txt, sort_key in col_config:
            self.tree.heading(col, text=txt, command=lambda c=col, k=sort_key: self._sort_by(c, k))
            self.tree.column(col, width=w)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        vsb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Button-2>", self._context_menu)
        self.tree.bind("<Button-3>", self._context_menu)
        main_pane.add(tree_frame, weight=3)

        preview = ttk.LabelFrame(main_pane, text="Details", padding=4)
        self.preview = scrolledtext.ScrolledText(preview, wrap=tk.WORD, font=("JetBrains Mono", 9),
                                                  bg="#0f172a", fg="#f8fafc", height=10)
        self.preview.pack(fill=tk.BOTH, expand=True)
        main_pane.add(preview, weight=1)

        # ─────────────────────────────────────────────────────────────
        # Pagination
        # ─────────────────────────────────────────────────────────────
        pager = ttk.Frame(self, style="Main.TFrame")
        pager.pack(fill=tk.X, pady=(3, 0))

        ttk.Button(pager, text="◀", command=self._prev_page, width=2).pack(side=tk.LEFT)
        self.page_label = ttk.Label(pager, text="1/1", style="Muted.TLabel")
        self.page_label.pack(side=tk.LEFT, padx=4)
        ttk.Button(pager, text="▶", command=self._next_page, width=2).pack(side=tk.LEFT)
        self.total_label = ttk.Label(pager, text="", style="Muted.TLabel")
        self.total_label.pack(side=tk.RIGHT)

    # ═══════════════════════════════════════════════════════════════
    # PUBLIC API (called by app.py)
    # ═══════════════════════════════════════════════════════════════
    def on_show(self):
        if not self.index_dropdown["values"]:
            self.call("load_pinecone_indexes")

    def set_indexes(self, indexes: list, current: str):
        self.index_dropdown["values"] = indexes
        self.index_var.set(current if current in indexes else (indexes[0] if indexes else ""))

    def set_namespaces(self, namespaces: list):
        self.namespace_dropdown["values"] = namespaces
        if namespaces:
            self.namespace_var.set(namespaces[0])

    def set_stats(self, stats: dict):
        for key in ("vectors", "dimension", "metric"):
            val = stats.get(key, "—")
            if key == "vectors" and isinstance(val, int):
                val = f"{val:,}"
            self.stat_labels[key].configure(text=str(val))

    def populate(self, vectors: list):
        self.all_vectors = vectors
        self.current_page = 1
        self._render_page()
        self.total_label.configure(text=f"{len(vectors):,} vectors")

    def get_selected_ids(self) -> list:
        return list(self.tree.selection())

    # ═══════════════════════════════════════════════════════════════
    # DIALOGS - Full SDK exposure
    # ═══════════════════════════════════════════════════════════════

    def _similarity_search_dialog(self):
        """Query similar vectors using SDK query() with all parameters."""
        d = tk.Toplevel(self.winfo_toplevel())
        d.title("Similarity Search")
        d.geometry("550x420")
        d.configure(bg=ModernTheme.COLORS["bg_main"])

        ttk.Label(d, text="Query Method:").pack(anchor="w", padx=10, pady=(10, 0))
        method_var = tk.StringVar(value="text")
        mf = ttk.Frame(d)
        mf.pack(fill=tk.X, padx=10)
        ttk.Radiobutton(mf, text="Text (auto-embed)", variable=method_var, value="text").pack(side=tk.LEFT)
        ttk.Radiobutton(mf, text="Vector ID", variable=method_var, value="id").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(mf, text="Raw vector", variable=method_var, value="vector").pack(side=tk.LEFT)

        ttk.Label(d, text="Query Input:").pack(anchor="w", padx=10, pady=(8, 0))
        query_text = scrolledtext.ScrolledText(d, height=5, font=("JetBrains Mono", 9),
                                               bg=ModernTheme.COLORS["bg_panel"], fg=ModernTheme.COLORS["text_main"])
        query_text.pack(fill=tk.X, padx=10, pady=4)

        pf = ttk.Frame(d)
        pf.pack(fill=tk.X, padx=10, pady=4)

        ttk.Label(pf, text="Top K:").grid(row=0, column=0, sticky="e")
        topk_var = tk.IntVar(value=10)
        ttk.Spinbox(pf, from_=1, to=10000, textvariable=topk_var, width=8).grid(row=0, column=1, sticky="w", padx=4)

        ttk.Label(pf, text="Namespace:").grid(row=0, column=2, sticky="e", padx=(10, 0))
        ns_var = tk.StringVar(value=self.namespace_var.get())
        ttk.Entry(pf, textvariable=ns_var, width=15).grid(row=0, column=3, sticky="w", padx=4)

        inc_meta = tk.BooleanVar(value=True)
        inc_vals = tk.BooleanVar(value=False)
        ttk.Checkbutton(pf, text="Include metadata", variable=inc_meta).grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(pf, text="Include values", variable=inc_vals).grid(row=1, column=2, columnspan=2, sticky="w")

        ttk.Label(d, text="Metadata Filter (JSON, optional):").pack(anchor="w", padx=10, pady=(4, 0))
        filter_text = scrolledtext.ScrolledText(d, height=3, font=("JetBrains Mono", 9),
                                                bg=ModernTheme.COLORS["bg_panel"], fg=ModernTheme.COLORS["text_main"])
        filter_text.pack(fill=tk.X, padx=10, pady=4)
        filter_text.insert("1.0", "{}")

        def do_query():
            q = query_text.get("1.0", tk.END).strip()
            if not q:
                return
            try:
                flt = json.loads(filter_text.get("1.0", tk.END).strip()) or None
            except:
                flt = None
            d.destroy()
            self.call("similarity_search", {
                "method": method_var.get(),
                "query": q,
                "top_k": topk_var.get(),
                "namespace": ns_var.get() or None,
                "include_metadata": inc_meta.get(),
                "include_values": inc_vals.get(),
                "filter": flt if flt != {} else None
            })

        ttk.Button(d, text="Search", command=do_query).pack(pady=10)

    def _metadata_filter_dialog(self):
        """Use fetch_by_metadata() SDK method - no embedding required."""
        d = tk.Toplevel(self.winfo_toplevel())
        d.title("Metadata Filter (fetch_by_metadata)")
        d.geometry("550x400")
        d.configure(bg=ModernTheme.COLORS["bg_main"])

        ttk.Label(d, text="Metadata Filter (JSON):").pack(anchor="w", padx=10, pady=10)

        txt = scrolledtext.ScrolledText(d, height=15, font=("JetBrains Mono", 9),
                                        bg=ModernTheme.COLORS["bg_panel"], fg=ModernTheme.COLORS["text_main"])
        txt.pack(fill=tk.BOTH, expand=True, padx=10)

        examples = '''# Examples:
# Simple: {"source": "plaud"}
# Equality: {"language": {"$eq": "en"}}
# Numeric: {"duration": {"$gt": 60}}
# And: {"$and": [{"source": "plaud"}, {"processed": true}]}
# In list: {"type": {"$in": ["call", "meeting"]}}

{"source": "plaud"}'''
        txt.insert("1.0", examples)

        pf = ttk.Frame(d)
        pf.pack(fill=tk.X, padx=10, pady=6)
        ttk.Label(pf, text="Limit:").pack(side=tk.LEFT)
        limit_var = tk.IntVar(value=100)
        ttk.Spinbox(pf, from_=1, to=1000, textvariable=limit_var, width=8).pack(side=tk.LEFT, padx=4)

        def do_search():
            raw = txt.get("1.0", tk.END)
            lines = [l for l in raw.split("\n") if not l.strip().startswith("#")]
            try:
                flt = json.loads("\n".join(lines))
            except json.JSONDecodeError as e:
                messagebox.showerror("JSON Error", str(e))
                return
            d.destroy()
            self.call("fetch_by_metadata", {"filter": flt, "limit": limit_var.get()})

        ttk.Button(d, text="Search", command=do_search).pack(pady=8)

    def _query_all_namespaces_dialog(self):
        """Use query_namespaces() SDK method - parallel search across all namespaces."""
        d = tk.Toplevel(self.winfo_toplevel())
        d.title("Query All Namespaces")
        d.geometry("500x350")
        d.configure(bg=ModernTheme.COLORS["bg_main"])

        ttk.Label(d, text="Search text (will be embedded):").pack(anchor="w", padx=10, pady=10)
        query_entry = ttk.Entry(d, width=60)
        query_entry.pack(fill=tk.X, padx=10)

        pf = ttk.Frame(d)
        pf.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(pf, text="Top K per NS:").pack(side=tk.LEFT)
        topk_var = tk.IntVar(value=10)
        ttk.Spinbox(pf, from_=1, to=100, textvariable=topk_var, width=6).pack(side=tk.LEFT, padx=4)

        ttk.Label(d, text="Filter (JSON, optional):").pack(anchor="w", padx=10)
        filter_txt = scrolledtext.ScrolledText(d, height=4, font=("JetBrains Mono", 9),
                                               bg=ModernTheme.COLORS["bg_panel"], fg=ModernTheme.COLORS["text_main"])
        filter_txt.pack(fill=tk.X, padx=10)
        filter_txt.insert("1.0", "{}")

        def do_query():
            q = query_entry.get().strip()
            if not q:
                return
            try:
                flt = json.loads(filter_txt.get("1.0", tk.END).strip()) or None
            except:
                flt = None
            d.destroy()
            self.call("query_all_namespaces", {"query": q, "top_k": topk_var.get(), "filter": flt if flt != {} else None})

        ttk.Button(d, text="Search All Namespaces", command=do_query).pack(pady=10)

    def _fetch_by_id_dialog(self):
        """Fetch specific vectors by ID using SDK fetch()."""
        ids = simpledialog.askstring("Fetch by ID", "Vector ID(s), comma-separated:")
        if ids:
            self.call("fetch_by_ids", [x.strip() for x in ids.split(",") if x.strip()])

    def _upsert_dialog(self):
        """Full upsert dialog with auto-embedding option."""
        d = tk.Toplevel(self.winfo_toplevel())
        d.title("Upsert Vector")
        d.geometry("600x500")
        d.configure(bg=ModernTheme.COLORS["bg_main"])

        f = ttk.Frame(d)
        f.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(f, text="Vector ID:").grid(row=0, column=0, sticky="w")
        id_entry = ttk.Entry(f, width=50)
        id_entry.grid(row=0, column=1, sticky="ew", pady=2)
        f.columnconfigure(1, weight=1)

        ttk.Label(f, text="Metadata (JSON):").grid(row=1, column=0, sticky="nw", pady=(8, 0))
        meta_txt = scrolledtext.ScrolledText(f, height=12, font=("JetBrains Mono", 9),
                                             bg=ModernTheme.COLORS["bg_panel"], fg=ModernTheme.COLORS["text_main"])
        meta_txt.grid(row=1, column=1, sticky="nsew", pady=2)
        f.rowconfigure(1, weight=1)

        default = {"title": "New Vector", "text": "Content for embedding goes here", "source": "manual"}
        meta_txt.insert("1.0", json.dumps(default, indent=2))

        ttk.Label(f, text="Embedding:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        embed_var = tk.StringVar(value="auto")
        ef = ttk.Frame(f)
        ef.grid(row=2, column=1, sticky="w")
        ttk.Radiobutton(ef, text="Auto from 'text' field", variable=embed_var, value="auto").pack(side=tk.LEFT)
        ttk.Radiobutton(ef, text="Manual (below)", variable=embed_var, value="manual").pack(side=tk.LEFT, padx=10)

        ttk.Label(f, text="Manual embedding (comma-sep):").grid(row=3, column=0, sticky="nw")
        embed_txt = scrolledtext.ScrolledText(f, height=3, font=("JetBrains Mono", 9),
                                              bg=ModernTheme.COLORS["bg_panel"], fg=ModernTheme.COLORS["text_main"])
        embed_txt.grid(row=3, column=1, sticky="ew", pady=2)

        def do_upsert():
            vec_id = id_entry.get().strip()
            if not vec_id:
                messagebox.showerror("Error", "Vector ID required")
                return
            try:
                metadata = json.loads(meta_txt.get("1.0", tk.END))
            except json.JSONDecodeError as e:
                messagebox.showerror("JSON Error", str(e))
                return

            d.destroy()
            self.call("upsert_vector", {
                "id": vec_id,
                "metadata": metadata,
                "embed_mode": embed_var.get(),
                "manual_embedding": embed_txt.get("1.0", tk.END).strip() if embed_var.get() == "manual" else None,
                "namespace": self.namespace_var.get() or None
            })

        ttk.Button(d, text="Upsert", command=do_upsert).pack(pady=10)

    def _edit_metadata_dialog(self):
        """Edit metadata of selected vector using SDK update()."""
        selected = self.get_selected_ids()
        if not selected:
            messagebox.showwarning("No Selection", "Select a vector to edit")
            return

        vec_id = selected[0]
        vec = next((v for v in self.all_vectors if v["id"] == vec_id), None)
        if not vec:
            return

        d = tk.Toplevel(self.winfo_toplevel())
        d.title(f"Edit: {vec_id[:20]}...")
        d.geometry("550x400")
        d.configure(bg=ModernTheme.COLORS["bg_main"])

        ttk.Label(d, text=f"Editing: {vec_id}").pack(anchor="w", padx=10, pady=10)

        txt = scrolledtext.ScrolledText(d, height=18, font=("JetBrains Mono", 9),
                                        bg=ModernTheme.COLORS["bg_panel"], fg=ModernTheme.COLORS["text_main"])
        txt.pack(fill=tk.BOTH, expand=True, padx=10)
        txt.insert("1.0", json.dumps(vec.get("metadata", {}), indent=2))

        def do_update():
            try:
                new_meta = json.loads(txt.get("1.0", tk.END))
            except json.JSONDecodeError as e:
                messagebox.showerror("JSON Error", str(e))
                return
            d.destroy()
            self.call("update_metadata", {"id": vec_id, "metadata": new_meta})

        ttk.Button(d, text="Update Metadata", command=do_update).pack(pady=10)

    def _delete_dialog(self):
        """Delete vectors with multiple options."""
        selected = self.get_selected_ids()

        d = tk.Toplevel(self.winfo_toplevel())
        d.title("Delete Vectors")
        d.geometry("450x300")
        d.configure(bg=ModernTheme.COLORS["bg_main"])

        mode_var = tk.StringVar(value="selected")
        ttk.Radiobutton(d, text=f"Delete selected ({len(selected)})", variable=mode_var, value="selected").pack(anchor="w", padx=10, pady=(10, 2))
        ttk.Radiobutton(d, text="Delete by filter (metadata)", variable=mode_var, value="filter").pack(anchor="w", padx=10, pady=2)
        ttk.Radiobutton(d, text="Delete ALL in namespace (⚠️)", variable=mode_var, value="all").pack(anchor="w", padx=10, pady=2)

        ttk.Label(d, text="Filter (for filter mode):").pack(anchor="w", padx=10, pady=(10, 0))
        filter_txt = scrolledtext.ScrolledText(d, height=5, font=("JetBrains Mono", 9),
                                               bg=ModernTheme.COLORS["bg_panel"], fg=ModernTheme.COLORS["text_main"])
        filter_txt.pack(fill=tk.X, padx=10)
        filter_txt.insert("1.0", '{"source": "manual"}')

        def do_delete():
            mode = mode_var.get()
            if mode == "selected":
                if not selected:
                    messagebox.showwarning("No Selection", "No vectors selected")
                    return
                if not messagebox.askyesno("Confirm", f"Delete {len(selected)} vector(s)?"):
                    return
                d.destroy()
                self.call("delete_vectors", {"ids": selected})
            elif mode == "filter":
                try:
                    flt = json.loads(filter_txt.get("1.0", tk.END))
                except:
                    messagebox.showerror("Error", "Invalid JSON filter")
                    return
                if not messagebox.askyesno("Confirm", f"Delete vectors matching filter?"):
                    return
                d.destroy()
                self.call("delete_vectors", {"filter": flt})
            else:
                if not messagebox.askyesno("⚠️ Danger", "Delete ALL vectors in namespace? This cannot be undone!"):
                    return
                d.destroy()
                self.call("delete_vectors", {"delete_all": True})

        ttk.Button(d, text="Delete", command=do_delete).pack(pady=10)

    def _create_namespace_dialog(self):
        """Create a new namespace using SDK create_namespace()."""
        name = simpledialog.askstring("Create Namespace", "New namespace name:")
        if name:
            self.call("create_namespace", name.strip())

    def _delete_namespace_dialog(self):
        """Delete a namespace using SDK delete_namespace()."""
        ns = self.namespace_var.get()
        if not ns:
            messagebox.showwarning("No Namespace", "Select a namespace first")
            return
        if messagebox.askyesno("Confirm", f"Delete namespace '{ns}' and ALL its vectors?"):
            self.call("delete_namespace", ns)

    def _list_namespaces_dialog(self):
        """Show all namespaces with vector counts."""
        self.call("list_namespaces_detail")

    def _bulk_import_dialog(self):
        """Import vectors from JSON/CSV with optional auto-embedding."""
        d = tk.Toplevel(self.winfo_toplevel())
        d.title("Bulk Import")
        d.geometry("650x550")
        d.configure(bg=ModernTheme.COLORS["bg_main"])

        # File selection
        file_frame = ttk.Frame(d)
        file_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(file_frame, text="File:").pack(side=tk.LEFT)
        file_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=file_var, width=50)
        file_entry.pack(side=tk.LEFT, padx=4)
        
        def browse():
            path = filedialog.askopenfilename(filetypes=[("JSON/CSV", "*.json *.csv"), ("JSON", "*.json"), ("CSV", "*.csv")])
            if path:
                file_var.set(path)
                preview_file(path)
        
        ttk.Button(file_frame, text="Browse", command=browse).pack(side=tk.LEFT)

        # Mode selection
        mode_frame = ttk.LabelFrame(d, text="Embedding Mode", padding=8)
        mode_frame.pack(fill=tk.X, padx=10, pady=6)
        
        mode_var = tk.StringVar(value="auto")
        ttk.Radiobutton(mode_frame, text="Auto-embed text fields (uses Gemini)", variable=mode_var, value="auto").pack(anchor="w")
        ttk.Radiobutton(mode_frame, text="Use existing embeddings (values/embedding field)", variable=mode_var, value="existing").pack(anchor="w")

        # Text field for auto-embed
        field_frame = ttk.Frame(mode_frame)
        field_frame.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(field_frame, text="Text field to embed:").pack(side=tk.LEFT)
        text_field_var = tk.StringVar(value="text")
        text_field_entry = ttk.Entry(field_frame, textvariable=text_field_var, width=20)
        text_field_entry.pack(side=tk.LEFT, padx=4)
        ttk.Label(field_frame, text="(e.g., text, content, transcript)", style="Muted.TLabel").pack(side=tk.LEFT)

        # ID field
        id_frame = ttk.Frame(mode_frame)
        id_frame.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(id_frame, text="ID field:").pack(side=tk.LEFT)
        id_field_var = tk.StringVar(value="id")
        ttk.Entry(id_frame, textvariable=id_field_var, width=20).pack(side=tk.LEFT, padx=4)
        ttk.Label(id_frame, text="(auto-generate if blank)", style="Muted.TLabel").pack(side=tk.LEFT)

        # Namespace
        ns_frame = ttk.Frame(d)
        ns_frame.pack(fill=tk.X, padx=10, pady=6)
        ttk.Label(ns_frame, text="Target namespace:").pack(side=tk.LEFT)
        ns_var = tk.StringVar(value=self.namespace_var.get())
        ttk.Entry(ns_frame, textvariable=ns_var, width=20).pack(side=tk.LEFT, padx=4)

        # Preview
        preview_frame = ttk.LabelFrame(d, text="File Preview (first 5 items)", padding=4)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        
        preview_txt = scrolledtext.ScrolledText(preview_frame, height=12, font=("JetBrains Mono", 9),
                                                 bg=ModernTheme.COLORS["bg_panel"], fg=ModernTheme.COLORS["text_main"])
        preview_txt.pack(fill=tk.BOTH, expand=True)

        def preview_file(path):
            preview_txt.delete("1.0", tk.END)
            try:
                if path.endswith(".csv"):
                    import csv
                    with open(path, 'r') as f:
                        reader = csv.DictReader(f)
                        rows = [row for _, row in zip(range(5), reader)]
                    if rows:
                        # Suggest fields
                        fields = list(rows[0].keys())
                        if "text" in fields:
                            text_field_var.set("text")
                        elif "content" in fields:
                            text_field_var.set("content")
                        elif "transcript" in fields:
                            text_field_var.set("transcript")
                        preview_txt.insert("1.0", f"CSV with {len(fields)} columns: {', '.join(fields)}\n\n")
                        preview_txt.insert(tk.END, json.dumps(rows[:3], indent=2))
                else:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    items = data if isinstance(data, list) else [data]
                    if items:
                        fields = list(items[0].keys()) if isinstance(items[0], dict) else []
                        # Auto-detect mode
                        if "values" in fields or "embedding" in fields:
                            mode_var.set("existing")
                            preview_txt.insert("1.0", "✓ Found existing embeddings\n\n")
                        else:
                            mode_var.set("auto")
                            if "text" in fields:
                                text_field_var.set("text")
                            elif "content" in fields:
                                text_field_var.set("content")
                            preview_txt.insert("1.0", f"Fields: {', '.join(fields)}\n\n")
                    preview_txt.insert(tk.END, json.dumps(items[:3], indent=2, default=str))
            except Exception as e:
                preview_txt.insert("1.0", f"Error reading file: {e}")

        # Format help
        help_txt = ttk.Label(d, text="Supported: JSON array of objects, CSV with headers. Auto-embed uses Gemini text-embedding-004.", 
                             style="Muted.TLabel", wraplength=600)
        help_txt.pack(padx=10, pady=(0, 6))

        # Buttons
        btn_frame = ttk.Frame(d)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        def do_import():
            path = file_var.get()
            if not path:
                messagebox.showwarning("No File", "Select a file first")
                return
            d.destroy()
            self.call("bulk_import", {
                "path": path,
                "mode": mode_var.get(),
                "text_field": text_field_var.get(),
                "id_field": id_field_var.get(),
                "namespace": ns_var.get() or None
            })

        ttk.Button(btn_frame, text="Import", command=do_import).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_frame, text="Cancel", command=d.destroy).pack(side=tk.RIGHT)

    def _export_dialog(self):
        """Export vectors to JSON."""
        if not self.all_vectors:
            messagebox.showwarning("No Data", "No vectors to export")
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if path:
            with open(path, "w") as f:
                json.dump(self.all_vectors, f, indent=2, default=str)
            messagebox.showinfo("Exported", f"Saved {len(self.all_vectors)} vectors")

    def _context_menu(self, event):
        """Right-click context menu."""
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="View Details", command=lambda: self._show_selected_detail())
        menu.add_command(label="Copy ID", command=lambda: self._copy_selected_id())
        menu.add_separator()
        menu.add_command(label="Edit Metadata", command=self._edit_metadata_dialog)
        menu.add_command(label="Delete", command=self._delete_dialog)
        menu.tk_popup(event.x_root, event.y_root)

    def _show_selected_detail(self):
        selected = self.get_selected_ids()
        if selected:
            vec = next((v for v in self.all_vectors if v["id"] == selected[0]), None)
            if vec:
                d = tk.Toplevel(self.winfo_toplevel())
                d.title(f"Vector: {selected[0][:30]}")
                d.geometry("600x500")
                txt = scrolledtext.ScrolledText(d, wrap=tk.WORD, font=("JetBrains Mono", 9))
                txt.pack(fill=tk.BOTH, expand=True)
                txt.insert("1.0", json.dumps(vec, indent=2, default=str))

    def _copy_selected_id(self):
        selected = self.get_selected_ids()
        if selected:
            self.clipboard_clear()
            self.clipboard_append(selected[0])

    # ═══════════════════════════════════════════════════════════════
    # PAGINATION & FILTERING
    # ═══════════════════════════════════════════════════════════════
    def _render_page(self):
        self.tree.delete(*self.tree.get_children())
        query = self.filter_var.get().lower()
        filtered = [v for v in self.all_vectors if query in json.dumps(v).lower()] if query else self.all_vectors[:]

        # Apply current sort
        def sort_key(v):
            val = v.get(self.sort_column, "")
            if val is None:
                val = ""
            # Handle numeric sorting for field_count and duration
            if self.sort_column == "field_count":
                return int(val) if isinstance(val, (int, float)) else 0
            if self.sort_column == "duration":
                # Parse "M:SS" format
                if isinstance(val, str) and ":" in val:
                    parts = val.split(":")
                    try:
                        return int(parts[0]) * 60 + int(parts[1])
                    except:
                        return 0
                return 0
            return str(val).lower()

        filtered.sort(key=sort_key, reverse=self.sort_reverse)

        try:
            self.page_size = int(self.page_size_var.get())
        except:
            self.page_size = 100

        start = (self.current_page - 1) * self.page_size
        page_data = filtered[start:start + self.page_size]

        for vec in page_data:
            self.tree.insert("", tk.END, iid=vec["id"], values=(
                vec.get("short_id", vec["id"][:10] + "…"),
                vec.get("title", "—")[:25],
                vec.get("date", "—"),
                vec.get("duration", "—"),
                vec.get("tags", "—")[:20],
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

    def _sort_by(self, col: str, sort_key: str):
        """Handle column header click for sorting."""
        # Toggle direction if clicking same column
        if self.sort_column == sort_key:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_column = sort_key
            self.sort_reverse = True if sort_key == "date" else False

        # Update header indicators
        col_labels = {"id": "ID", "title": "Title", "date": "Date", "dur": "Dur", "tags": "Tags", "flds": "#"}
        col_keys = {"id": "id", "title": "title", "date": "date", "dur": "duration", "tags": "tags", "flds": "field_count"}
        for c, label in col_labels.items():
            if col_keys[c] == sort_key:
                arrow = "▼" if self.sort_reverse else "▲"
                self.tree.heading(c, text=f"{label} {arrow}")
            else:
                self.tree.heading(c, text=label)

        self.current_page = 1
        self._render_page()

    def _on_select(self, _event):
        selected = self.tree.selection()
        if not selected:
            return
        vec = next((v for v in self.all_vectors if v["id"] == selected[0]), None)
        if vec:
            self.preview.configure(state=tk.NORMAL)
            self.preview.delete("1.0", tk.END)
            self.preview.insert("1.0", json.dumps(vec.get("metadata", vec), indent=2, default=str))
            self.preview.configure(state=tk.DISABLED)
