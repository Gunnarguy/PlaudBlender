import tkinter as tk
from tkinter import ttk
from gui.views.base import BaseView


class SearchView(BaseView):
    """
    Search view with ultra-granular, clearly-labeled search actions.
    
    Each button corresponds to a specific, well-documented action:
    - 🌐 Search All: Cross-namespace parallel search
    - 📄 Full Text: Search transcripts only
    - 📝 Summaries: Search AI summaries only
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
            text="Search your transcripts using natural language",
            font=("Inter", 9),
        ).pack(anchor='w')

        # ─────────────────────────────────────────────────────────────
        # Search Input
        # ─────────────────────────────────────────────────────────────
        input_frame = ttk.Frame(self, style="Main.TFrame")
        input_frame.pack(fill=tk.X, pady=(0, 8))

        self.query_var = tk.StringVar()
        entry = ttk.Entry(
            input_frame, 
            textvariable=self.query_var, 
            font=("Inter", 11),
        )
        entry.pack(fill=tk.X, ipady=4)
        entry.bind('<Return>', lambda _: self._search_all())

        # ─────────────────────────────────────────────────────────────
        # Search Action Buttons - ULTRA GRANULAR
        # ─────────────────────────────────────────────────────────────
        actions_frame = ttk.Frame(self, style="Main.TFrame")
        actions_frame.pack(fill=tk.X, pady=(0, 8))
        
        # Row 1: Primary search actions
        btn_row1 = ttk.Frame(actions_frame, style="Main.TFrame")
        btn_row1.pack(fill=tk.X, pady=(0, 4))
        
        ttk.Button(
            btn_row1,
            text="🌐 Search All Namespaces",
            style="Accent.TButton",
            command=self._search_all,
            width=25,
        ).pack(side=tk.LEFT, padx=(0, 4))
        
        ttk.Button(
            btn_row1,
            text="📄 Search Full Text Only",
            command=self._search_full_text,
            width=25,
        ).pack(side=tk.LEFT, padx=(0, 4))
        
        ttk.Button(
            btn_row1,
            text="📝 Search Summaries Only",
            command=self._search_summaries,
            width=25,
        ).pack(side=tk.LEFT)

        # ─────────────────────────────────────────────────────────────
        # Options Row
        # ─────────────────────────────────────────────────────────────
        options_frame = ttk.Frame(self, style="Main.TFrame")
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
        
        # Include context toggle
        self.context_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame,
            text="Include text snippets",
            variable=self.context_var,
        ).pack(side=tk.LEFT)

        # ─────────────────────────────────────────────────────────────
        # Help Text
        # ─────────────────────────────────────────────────────────────
        help_frame = ttk.Frame(self, style="Main.TFrame")
        help_frame.pack(fill=tk.X, pady=(0, 8))
        
        help_text = (
            "💡 Tips:\n"
            "• 🌐 Search All: Searches both full transcripts AND summaries in parallel\n"
            "• 📄 Full Text: Best for finding specific quotes or passages\n"
            "• 📝 Summaries: Best for finding topics or themes"
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
        results_label = ttk.Label(self, text="Results", font=("Inter", 10, "bold"))
        results_label.pack(anchor='w', pady=(0, 4))
        
        self.results = tk.Text(
            self, 
            bg="#0f172a", 
            fg="#f8fafc", 
            insertbackground="#f8fafc",
            wrap=tk.WORD,
            padx=12,
            pady=12,
            font=("JetBrains Mono", 10),
        )
        self.results.pack(fill=tk.BOTH, expand=True)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.results, command=self.results.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results.config(yscrollcommand=scrollbar.set)
    
    # ─────────────────────────────────────────────────────────────────
    # Search Action Methods - Each maps to ONE specific backend action
    # ─────────────────────────────────────────────────────────────────
    
    def _search_all(self):
        """🌐 SEARCH ALL NAMESPACES - Parallel cross-namespace search."""
        query = self.query_var.get()
        limit = int(self.limit_var.get())
        if query.strip():
            self.call('perform_cross_namespace_search', query, limit)
    
    def _search_full_text(self):
        """📄 SEARCH FULL TEXT - Search only the full_text namespace."""
        query = self.query_var.get()
        limit = int(self.limit_var.get())
        if query.strip():
            self.call('search_full_text', query, limit)
    
    def _search_summaries(self):
        """📝 SEARCH SUMMARIES - Search only the summaries namespace."""
        query = self.query_var.get()
        limit = int(self.limit_var.get())
        if query.strip():
            self.call('search_summaries', query, limit)

    # ─────────────────────────────────────────────────────────────────
    # Display Methods
    # ─────────────────────────────────────────────────────────────────

    def show_results(self, text):
        """Display search results in the text area."""
        self.results.delete('1.0', tk.END)
        self.results.insert('1.0', text)
