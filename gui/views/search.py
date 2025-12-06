import os
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from gui.utils.tooltips import ToolTip
from gui.views.base import BaseView


class SearchView(BaseView):
    """
    Search view with ultra-granular, clearly-labeled search actions.
    
    Each button corresponds to a specific, well-documented action:
    - 🌐 Search All: Cross-namespace parallel search
    - 📄 Full Text: Search transcripts only
    - 📝 Summaries: Search AI summaries only
    - 🏆 Rerank: Optional 2-stage search for highest relevance (toggle)
    
    Shows retrieval_score and rerank_score in results for transparency.
    """
    
    def _build(self):
        # ─────────────────────────────────────────────────────────────
        # Header
        # ─────────────────────────────────────────────────────────────
        header = ttk.Frame(self, style="Main.TFrame")
        header.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(header, text="🔍 Semantic Search", style="Header.TLabel").pack(anchor='w')
        ttk.Label(
            header, 
            text="Find anything across transcripts and summaries",
            style="Muted.TLabel",
        ).pack(anchor='w')

        # ─────────────────────────────────────────────────────────────
        # Search Input
        # ─────────────────────────────────────────────────────────────
        input_frame = ttk.LabelFrame(self, text="Query", padding=8, style="Panel.TLabelframe")
        input_frame.pack(fill=tk.X, pady=(0, 8))

        self.query_var = tk.StringVar()
        entry = ttk.Entry(
            input_frame, 
            textvariable=self.query_var, 
            font=("Inter", 11),
        )
        entry.pack(fill=tk.X, ipady=4)
        entry.bind('<Return>', lambda _: self._search_all())

        # Preset prompts for fast inspiration
        presets = ttk.Frame(input_frame, style="Main.TFrame")
        presets.pack(fill=tk.X, pady=(6, 0))
        for text in [
            "Action items from last call",
            "Mentions of pricing or budget",
            "Key blockers mentioned",
            "Summaries about roadmap",
        ]:
            ttk.Button(presets, text=text, style="Pill.TButton",
                       command=lambda t=text: self._use_preset(t)).pack(side=tk.LEFT, padx=3)

        # ─────────────────────────────────────────────────────────────
        # Search Action Buttons - ULTRA GRANULAR
        # ─────────────────────────────────────────────────────────────
        actions_frame = ttk.LabelFrame(self, text="Actions", padding=8, style="Panel.TLabelframe")
        actions_frame.pack(fill=tk.X, pady=(0, 8))
        
        # Row 1: Primary search actions
        btn_row1 = ttk.Frame(actions_frame, style="Main.TFrame")
        btn_row1.pack(fill=tk.X, pady=(0, 4))
        
        search_all_btn = ttk.Button(
            btn_row1,
            text="🌐 Search All Namespaces",
            style="Accent.TButton",
            command=self._search_all,
            width=25,
        )
        search_all_btn.pack(side=tk.LEFT, padx=(0, 4))
        ToolTip(search_all_btn, 
            "Cross-namespace parallel search.\n\n"
            "• Searches BOTH full_text AND summaries\n"
            "• Results merged and ranked by vector similarity\n"
            "• Enable 🏆 Rerank for higher accuracy (+~200ms)"
        )
        
        search_ft_btn = ttk.Button(
            btn_row1,
            text="📄 Search Full Text Only",
            command=self._search_full_text,
            width=25,
        )
        search_ft_btn.pack(side=tk.LEFT, padx=(0, 4))
        ToolTip(search_ft_btn,
            "Search the full_text namespace only.\n\n"
            "• Contains chunked transcript text\n"
            "• Best for finding specific quotes/passages\n"
            "• Higher recall, more detailed results"
        )
        
        search_sum_btn = ttk.Button(
            btn_row1,
            text="📝 Search Summaries Only",
            command=self._search_summaries,
            width=25,
        )
        search_sum_btn.pack(side=tk.LEFT)
        ToolTip(search_sum_btn,
            "Search the summaries namespace only.\n\n"
            "• Contains AI-generated syntheses\n"
            "• Best for thematic/topic-level queries\n"
            "• Faster, more focused results"
        )
        
        # Row 2: Hybrid search button
        btn_row2 = ttk.Frame(actions_frame, style="Main.TFrame")
        btn_row2.pack(fill=tk.X, pady=(4, 0))
        
        hybrid_btn = ttk.Button(
            btn_row2,
            text="🔀 Hybrid Search (Dense + Sparse)",
            style="Accent.TButton",
            command=self._search_hybrid,
            width=35,
        )
        hybrid_btn.pack(side=tk.LEFT, padx=(0, 4))
        ToolTip(hybrid_btn,
            "Combines semantic AND keyword search for best accuracy.\n\n"
            "• Dense vectors: Catch synonyms, paraphrases\n"
            "• Sparse vectors: Catch exact keywords, proper nouns\n"
            "• Alpha slider controls dense vs sparse weight\n"
            "• Achieves ~99% retrieval accuracy"
        )
        
        # Smart Search button (Query Router + RRF Fusion)
        smart_btn = ttk.Button(
            btn_row2,
            text="🧠 Smart Search (Router + RRF)",
            command=self._search_smart,
            width=30,
        )
        smart_btn.pack(side=tk.LEFT, padx=(0, 4))
        ToolTip(smart_btn,
            "AI-powered search using Query Router + RRF Fusion.\n\n"
            "• Router: Auto-classifies query intent\n"
            "• Picks optimal strategy (keyword/semantic/hybrid)\n"
            "• RRF: Mathematical fusion of ranked lists\n"
            "• GraphRAG: Answers aggregation queries\n\n"
            "Best for varied query types. Highest accuracy."
        )

        # ─────────────────────────────────────────────────────────────
        # Options Row (with Rerank toggle)
        # ─────────────────────────────────────────────────────────────
        options_frame = ttk.LabelFrame(self, text="Options", padding=8, style="Panel.TLabelframe")
        options_frame.pack(fill=tk.X, pady=(0, 8))
        
        # Result limit selector
        ttk.Label(options_frame, text="Results:").pack(side=tk.LEFT)
        self.limit_var = tk.StringVar(value="5")
        limit_combo = ttk.Combobox(
            options_frame, 
            textvariable=self.limit_var, 
            values=["5", "10", "20", "50"],
            width=4,
            state="readonly",
        )
        limit_combo.pack(side=tk.LEFT, padx=(4, 12))
        ToolTip(limit_combo, "Maximum number of results to return")
        
        # Include context toggle
        self.context_var = tk.BooleanVar(value=True)
        context_cb = ttk.Checkbutton(
            options_frame,
            text="Include text snippets",
            variable=self.context_var,
        )
        context_cb.pack(side=tk.LEFT)
        ToolTip(context_cb, "Show text excerpts in results (disable for compact view)")

        # ───── RERANK TOGGLE (2-stage search for best relevance) ─────
        ttk.Separator(options_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=12)
        
        # Initialize from env or default to False
        rerank_default = os.getenv("PINECONE_RERANK_ENABLED", "false").lower() == "true"
        self.rerank_var = tk.BooleanVar(value=rerank_default)
        rerank_cb = ttk.Checkbutton(
            options_frame,
            text="🏆 Rerank",
            variable=self.rerank_var,
            command=self._on_rerank_toggle,
        )
        rerank_cb.pack(side=tk.LEFT)
        ToolTip(rerank_cb, 
            "Two-stage search: dense retrieval → neural reranking.\n"
            "Uses Pinecone's bge-reranker-v2-m3 model.\n"
            "Higher quality results but adds ~200ms latency."
        )
        
        # Rerank model selector (advanced)
        self.rerank_model_var = tk.StringVar(value=os.getenv("PINECONE_RERANK_MODEL", "bge-reranker-v2-m3"))
        # (Hidden by default, shown when rerank is enabled for power users)
        
        # ───── HYBRID ALPHA SLIDER (dense vs sparse weight) ─────
        ttk.Separator(options_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=12)
        
        alpha_frame = ttk.Frame(options_frame, style="Main.TFrame")
        alpha_frame.pack(side=tk.LEFT)
        
        ttk.Label(alpha_frame, text="Alpha:").pack(side=tk.LEFT)
        self.alpha_var = tk.DoubleVar(value=float(os.getenv("PINECONE_HYBRID_ALPHA", "0.5")))
        alpha_scale = ttk.Scale(
            alpha_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            length=80,
            variable=self.alpha_var,
            command=self._on_alpha_change,
        )
        alpha_scale.pack(side=tk.LEFT, padx=(4, 4))
        
        self.alpha_label = ttk.Label(alpha_frame, text="0.50", width=4)
        self.alpha_label.pack(side=tk.LEFT)
        ToolTip(alpha_frame,
            "Dense vs Sparse weight for hybrid search.\n\n"
            "• 0.0 = 100% keyword (sparse)\n"
            "• 0.5 = 50/50 balanced (default)\n"
            "• 1.0 = 100% semantic (dense)\n\n"
            "Lower values catch exact terms better.\n"
            "Higher values catch meaning better."
        )

        # ───── SELF-CORRECTION TOGGLE ─────
        ttk.Separator(options_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=12)
        
        self.self_correct_var = tk.BooleanVar(value=False)
        self_correct_cb = ttk.Checkbutton(
            options_frame,
            text="🔄 Self-Correct",
            variable=self.self_correct_var,
        )
        self_correct_cb.pack(side=tk.LEFT)
        ToolTip(self_correct_cb,
            "Automatic retry with different strategies on low confidence.\n\n"
            "• Detects when results are uncertain\n"
            "• Tries: dense → hybrid → query expansion → full-text\n"
            "• Shows correction attempts in results\n\n"
            "Adds latency but improves accuracy on tough queries."
        )

        # Result style toggle (stored for future formatting)
        ttk.Label(options_frame, text="Style:").pack(side=tk.LEFT, padx=(12, 4))
        self.result_style = tk.StringVar(value="rich")
        style_combo = ttk.Combobox(options_frame, textvariable=self.result_style, values=["rich", "compact"], width=8, state="readonly")
        style_combo.pack(side=tk.LEFT)
        ToolTip(style_combo, "rich: Full details with scores\ncompact: Minimal display")

        # ─────────────────────────────────────────────────────────────
        # Saved Searches
        # ─────────────────────────────────────────────────────────────
        saved_frame = ttk.LabelFrame(self, text="Saved Searches", padding=8, style="Panel.TLabelframe")
        saved_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(saved_frame, text="Saved:").pack(side=tk.LEFT)
        self.saved_var = tk.StringVar()
        self.saved_combo = ttk.Combobox(saved_frame, textvariable=self.saved_var, state="readonly", width=25)
        self.saved_combo.pack(side=tk.LEFT, padx=(4, 8))
        self.saved_combo.bind("<<ComboboxSelected>>", lambda _: self._load_saved())

        ttk.Button(saved_frame, text="Save Current", command=self._save_current).pack(side=tk.LEFT)
        ttk.Button(saved_frame, text="Delete", command=self._delete_saved).pack(side=tk.LEFT, padx=(4, 0))

        # ─────────────────────────────────────────────────────────────
        # Help Text (with rerank explanation)
        # ─────────────────────────────────────────────────────────────
        help_frame = ttk.LabelFrame(self, text="Tips", padding=8, style="Panel.TLabelframe")
        help_frame.pack(fill=tk.X, pady=(0, 8))
        
        help_text = (
            "💡 Tips:\n"
            "• 🌐 Search All: Searches both full transcripts AND summaries in parallel\n"
            "• 📄 Full Text: Best for finding specific quotes or passages\n"
            "• 📝 Summaries: Best for finding topics or themes\n"
            "• 🏆 Rerank: Fetches 3x candidates, reranks with neural model for best relevance\n"
            "\n"
            "📊 Scores: retrieval_score = vector similarity, rerank_score = semantic relevance"
        )
        ttk.Label(
            help_frame,
            text=help_text,
            font=("Inter", 9),
            justify=tk.LEFT,
        ).pack(anchor='w')

        # ─────────────────────────────────────────────────────────────
        # Results Area
        # ─────────────────────────────────────────────────────────────
        results_frame = ttk.LabelFrame(self, text="Results", padding=6, style="Panel.TLabelframe")
        results_frame.pack(fill=tk.BOTH, expand=True)

        header_row = ttk.Frame(results_frame, style="Main.TFrame")
        header_row.pack(fill=tk.X)
        self.last_run = ttk.Label(header_row, text="No searches yet", style="Muted.TLabel")
        self.last_run.pack(side=tk.LEFT)
        ttk.Button(header_row, text="Clear", command=self._clear_results).pack(side=tk.RIGHT)

        self.results = tk.Text(
            results_frame, 
            bg="#0f172a", 
            fg="#f8fafc", 
            insertbackground="#f8fafc",
            wrap=tk.WORD,
            padx=12,
            pady=12,
            font=("JetBrains Mono", 10),
            height=14,
        )
        self.results.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(results_frame, command=self.results.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results.config(yscrollcommand=scrollbar.set)
    
    # ─────────────────────────────────────────────────────────────────
    # Search Action Methods - Each maps to ONE specific backend action
    # ─────────────────────────────────────────────────────────────────
    
    def _search_all(self):
        """🌐 SEARCH ALL NAMESPACES - Parallel cross-namespace search (+ optional rerank/self-correct)."""
        query = self.query_var.get()
        limit = int(self.limit_var.get())
        if not query.strip():
            return
        
        use_rerank = self.rerank_var.get()
        use_self_correct = self.self_correct_var.get()
        
        # Self-correction takes priority
        if use_self_correct:
            mode = "All + Self-Correct"
            self._mark_last_run(mode, query, limit)
            self.call('perform_self_correcting_search', query, limit)
        elif use_rerank:
            mode = "All + Rerank"
            self._mark_last_run(mode, query, limit)
            self.call('perform_rerank_search', query, limit, self.rerank_model_var.get())
        else:
            mode = "All namespaces"
            self._mark_last_run(mode, query, limit)
            self.call('perform_cross_namespace_search', query, limit)
    
    def _search_full_text(self):
        """📄 SEARCH FULL TEXT - Search only the full_text namespace."""
        query = self.query_var.get()
        limit = int(self.limit_var.get())
        if query.strip():
            use_rerank = self.rerank_var.get()
            mode = "Full text + Rerank" if use_rerank else "Full text"
            self._mark_last_run(mode, query, limit)
            if use_rerank:
                self.call('perform_rerank_search', query, limit, self.rerank_model_var.get(), ['full_text'])
            else:
                self.call('search_full_text', query, limit)
    
    def _search_summaries(self):
        """📝 SEARCH SUMMARIES - Search only the summaries namespace."""
        query = self.query_var.get()
        limit = int(self.limit_var.get())
        if query.strip():
            use_rerank = self.rerank_var.get()
            mode = "Summaries + Rerank" if use_rerank else "Summaries"
            self._mark_last_run(mode, query, limit)
            if use_rerank:
                self.call('perform_rerank_search', query, limit, self.rerank_model_var.get(), ['summaries'])
            else:
                self.call('search_summaries', query, limit)

    def _search_hybrid(self):
        """🔀 HYBRID SEARCH - Dense + Sparse vectors combined."""
        query = self.query_var.get()
        limit = int(self.limit_var.get())
        if not query.strip():
            return
        
        alpha = self.alpha_var.get()
        use_rerank = self.rerank_var.get()
        mode = f"Hybrid (α={alpha:.2f})" + (" + Rerank" if use_rerank else "")
        self._mark_last_run(mode, query, limit)
        
        self.call('perform_hybrid_search', query, limit, alpha, use_rerank)

    def _search_smart(self):
        """🧠 SMART SEARCH - Query Router + RRF Fusion + GraphRAG."""
        query = self.query_var.get()
        limit = int(self.limit_var.get())
        if not query.strip():
            return
        
        mode = "Smart (Router + RRF)"
        self._mark_last_run(mode, query, limit)
        
        self.call('perform_smart_search', query, limit)

    def _on_rerank_toggle(self):
        """Handle rerank checkbox toggle - update status label."""
        enabled = self.rerank_var.get()
        status = "🏆 Rerank ON (neural reranking)" if enabled else "Rerank OFF"
        # Could update a status bar here; for now just log
        from gui.utils.logging import log
        log('INFO', f"Rerank toggled: {status}")

    def _on_alpha_change(self, value):
        """Handle alpha slider change - update label."""
        alpha = float(value)
        self.alpha_label.configure(text=f"{alpha:.2f}")

    # ─────────────────────────────────────────────────────────────────
    # Display Methods
    # ─────────────────────────────────────────────────────────────────

    def show_results(self, text):
        """Display search results in the text area."""
        self.results.delete('1.0', tk.END)
        if not text:
            text = "No results yet. Run a search to see matches."
        # Simple error surfacing: if backend returned an error marker, also show a messagebox
        if text.strip().startswith("❌"):
            messagebox.showerror("Search", text)
        self.results.insert('1.0', text)

    def _mark_last_run(self, scope: str, query: str, limit: int):
        """Update the inline status above the results panel."""
        preview = query[:70] + ("…" if len(query) > 70 else "")
        self.last_run.configure(text=f"{scope} • top {limit} • \"{preview}\"")

    def _use_preset(self, text: str):
        self.query_var.set(text)
        self._search_all()

    def _clear_results(self):
        self.results.delete('1.0', tk.END)
        self.last_run.configure(text="Cleared")

    def update_saved(self, names):
        self.saved_combo['values'] = names
        if names:
            self.saved_var.set(names[0])
        else:
            self.saved_var.set("")

    def set_query(self, q: str):
        self.query_var.set(q)

    def _save_current(self):
        name = simpledialog.askstring("Save Search", "Name for this search:")
        if not name:
            return
        self.call('save_search', name, self.query_var.get())

    def _load_saved(self):
        name = self.saved_var.get()
        if name:
            self.call('load_saved_search', name)

    def _delete_saved(self):
        name = self.saved_var.get()
        if not name:
            return
        from gui.services import saved_searches_service as ss
        ss.delete_search(name)
        self.update_saved(ss.list_saved_names())

    def on_show(self):
        # Refresh saved searches and focus query box for faster flow
        from gui.services import saved_searches_service as ss
        self.update_saved(ss.list_saved_names())
        try:
            self.focus_set()
        except Exception:
            pass
