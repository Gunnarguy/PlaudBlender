import os
import tkinter as tk
from tkinter import ttk, messagebox
from gui.views.base import BaseView
from gui.state import state
from gui.utils.tooltips import ToolTip
from gui.utils.async_tasks import run_async


class SettingsView(BaseView):
    def on_show(self):  # pragma: no cover - UI
        self._refresh_status()
        self._load_index_status()
        self._update_preflight()

    def _build(self):
        container = ttk.Frame(self, style="Main.TFrame")
        container.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Hero
        hero = ttk.Frame(container, padding=(0, 4))
        hero.pack(fill=tk.X)
        ttk.Label(hero, text="Settings", style="Header.TLabel").pack(anchor="w")
        ttk.Label(hero, text="Keep keys, provider, and index in one clean view.", style="Muted.TLabel").pack(anchor="w")

        # Status badges row
        status_row = ttk.Frame(container, padding=(0, 4))
        status_row.pack(fill=tk.X, pady=(2, 6))
        self.plaud_status = ttk.Label(status_row, text="Plaud: —", style="Muted.TLabel")
        self.plaud_status.pack(side=tk.LEFT, padx=(0, 12))
        self.pinecone_status = ttk.Label(status_row, text="Pinecone: —", style="Muted.TLabel")
        self.pinecone_status.pack(side=tk.LEFT, padx=(0, 12))
        self.gemini_status = ttk.Label(status_row, text="LLM: —", style="Muted.TLabel")
        self.gemini_status.pack(side=tk.LEFT, padx=(0, 12))

        # Preflight summary card
        preflight = ttk.LabelFrame(container, text="Preflight", padding=10, style="Panel.TFrame")
        preflight.pack(fill=tk.X, pady=(0, 8))
        self.preflight_label = ttk.Label(preflight, text="Running preflight...", style="Muted.TLabel", justify="left")
        self.preflight_label.pack(anchor="w")

        # Quick actions
        actions_row = ttk.Frame(container, padding=(0, 4))
        actions_row.pack(fill=tk.X, pady=(0, 8))
        btn_token = ttk.Button(actions_row, text="✅ Validate Plaud token", command=self._validate_token)
        btn_token.pack(side=tk.LEFT, padx=(0, 6))
        ToolTip(btn_token, "Call Plaud /users/current to confirm access token; auto-refresh if possible")
        btn_consent = ttk.Button(actions_row, text="🔗 Copy consent URL", command=self._copy_consent_url)
        btn_consent.pack(side=tk.LEFT, padx=6)
        ToolTip(btn_consent, "Copy the Plaud OAuth authorize link to clipboard")
        btn_index = ttk.Button(actions_row, text="📦 Refresh index status", command=self._load_index_status)
        btn_index.pack(side=tk.LEFT, padx=6)
        ToolTip(btn_index, "Re-check Pinecone index dimension & namespaces")
        btn_env = ttk.Button(actions_row, text="📝 Open .env", command=self._open_env)
        btn_env.pack(side=tk.LEFT, padx=6)
        ToolTip(btn_env, "Open .env in your default editor")
        btn_restart = ttk.Button(actions_row, text="🔄 Save & Restart", command=self._restart_app)
        btn_restart.pack(side=tk.LEFT, padx=6)
        ToolTip(btn_restart, "Save settings, close, and restart the app to reload environment")

        # Two-column layout
        grid = ttk.Frame(container, style="Main.TFrame")
        grid.pack(fill=tk.BOTH, expand=True)
        grid.columnconfigure(0, weight=1, uniform="col")
        grid.columnconfigure(1, weight=1, uniform="col")

        # Left: Connections
        conn_card = ttk.LabelFrame(grid, text="Connections", padding=10, style="Panel.TFrame")
        conn_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 6))

        self.env_entries = {}

        def add_field(frame, row, label, key, width=42, secret=False):
            ttk.Label(frame, text=label, style="Muted.TLabel").grid(row=row, column=0, sticky="w", pady=3, padx=(0, 6))
            var = tk.StringVar(value=os.getenv(key, ""))
            entry = ttk.Entry(frame, textvariable=var, width=width, show="*" if secret else "")
            entry.grid(row=row, column=1, sticky="ew", pady=3)
            self.env_entries[key] = var

        add_field(conn_card, 0, "Plaud Client ID", "PLAUD_CLIENT_ID")
        add_field(conn_card, 1, "Plaud Client Secret", "PLAUD_CLIENT_SECRET", secret=True)
        add_field(conn_card, 2, "Plaud Redirect URI", "PLAUD_REDIRECT_URI")
        add_field(conn_card, 3, "Pinecone API Key", "PINECONE_API_KEY", secret=True)
        add_field(conn_card, 4, "Default Index Name", "PINECONE_INDEX_NAME")
        conn_card.columnconfigure(1, weight=1)
        ttk.Label(conn_card, text="Saved to .env; keys stay local.", style="Muted.TLabel").grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 0))

        # Right: LLM & Embedding
        embed_card = ttk.LabelFrame(grid, text="LLM & Embeddings", padding=10, style="Panel.TFrame")
        embed_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=(0, 6))

        # Provider + models
        row0 = ttk.Frame(embed_card)
        row0.pack(fill=tk.X, pady=2)
        ttk.Label(row0, text="Provider", width=12, anchor="w").pack(side=tk.LEFT)
        self.provider_var = tk.StringVar(value=os.getenv("AI_PROVIDER", "google"))
        self.provider_combo = ttk.Combobox(row0, textvariable=self.provider_var, values=["google", "openai"], state="readonly", width=12)
        self.provider_combo.pack(side=tk.LEFT, padx=(4, 10))
        self.provider_combo.bind("<<ComboboxSelected>>", self._on_provider_change)
        self.env_entries["AI_PROVIDER"] = self.provider_var

        # Gemini model dropdown
        row1 = ttk.Frame(embed_card)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="Gemini model", width=12, anchor="w").pack(side=tk.LEFT)
        gemini_models = ["gemini-embedding-001", "models/text-embedding-004"]
        gemini_default = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
        if gemini_default not in gemini_models:
            gemini_models.append(gemini_default)
        self.model_var = tk.StringVar(value=gemini_default)
        self.gemini_combo = ttk.Combobox(row1, textvariable=self.model_var, values=gemini_models, state="readonly", width=26)
        self.gemini_combo.pack(side=tk.LEFT, padx=(4, 6))
        self.gemini_combo.bind("<<ComboboxSelected>>", self._on_model_change)
        self.env_entries["GEMINI_EMBEDDING_MODEL"] = self.model_var

        # Gemini dimension (configurable 128-3072, recommended: 768, 1536, 3072)
        row1b = ttk.Frame(embed_card)
        row1b.pack(fill=tk.X, pady=2)
        ttk.Label(row1b, text="Gemini dim", width=12, anchor="w").pack(side=tk.LEFT)
        self.gemini_dim_var = tk.StringVar(value=os.getenv("GEMINI_EMBEDDING_DIM", "768"))
        gemini_dim_combo = ttk.Combobox(row1b, textvariable=self.gemini_dim_var, 
                                         values=["256", "512", "768", "1536", "3072"], state="readonly", width=8)
        gemini_dim_combo.pack(side=tk.LEFT, padx=(4, 6))
        self.env_entries["GEMINI_EMBEDDING_DIM"] = self.gemini_dim_var
        ttk.Label(row1b, text="128-3072 supported; 768 recommended", style="Muted.TLabel").pack(side=tk.LEFT, padx=(4, 0))

        # Gemini task type
        row1c = ttk.Frame(embed_card)
        row1c.pack(fill=tk.X, pady=2)
        ttk.Label(row1c, text="Task type", width=12, anchor="w").pack(side=tk.LEFT)
        task_types = [
            "RETRIEVAL_DOCUMENT",
            "RETRIEVAL_QUERY", 
            "SEMANTIC_SIMILARITY",
            "CLUSTERING",
            "CLASSIFICATION",
            "QUESTION_ANSWERING",
        ]
        self.task_type_var = tk.StringVar(value=os.getenv("GEMINI_TASK_TYPE", "RETRIEVAL_DOCUMENT"))
        ttk.Combobox(row1c, textvariable=self.task_type_var, values=task_types, state="readonly", width=22).pack(side=tk.LEFT, padx=(4, 6))
        self.env_entries["GEMINI_TASK_TYPE"] = self.task_type_var
        ttk.Label(row1c, text="Optimizes embeddings for use case", style="Muted.TLabel").pack(side=tk.LEFT, padx=(4, 0))

        # Hint for Gemini
        self.gemini_hint = ttk.Label(embed_card, text="Free tier · Native 3072d, configurable", style="Muted.TLabel", foreground="#27ae60")
        self.gemini_hint.pack(anchor="w", padx=(100, 0))

        ttk.Separator(embed_card, orient="horizontal").pack(fill=tk.X, pady=6)

        # OpenAI model dropdown
        row2 = ttk.Frame(embed_card)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="OpenAI model", width=12, anchor="w").pack(side=tk.LEFT)
        openai_models = [
            "text-embedding-3-large",
            "text-embedding-3-small",
            "text-embedding-ada-002",
        ]
        openai_default = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
        if openai_default not in openai_models:
            openai_models.append(openai_default)
        self.openai_model_var = tk.StringVar(value=openai_default)
        self.openai_combo = ttk.Combobox(row2, textvariable=self.openai_model_var, values=openai_models, state="readonly", width=26)
        self.openai_combo.pack(side=tk.LEFT, padx=(4, 6))
        self.openai_combo.bind("<<ComboboxSelected>>", self._on_model_change)
        self.env_entries["OPENAI_EMBEDDING_MODEL"] = self.openai_model_var

        # OpenAI dimension (configurable for 3-large/3-small only)
        row2b = ttk.Frame(embed_card)
        row2b.pack(fill=tk.X, pady=2)
        ttk.Label(row2b, text="OpenAI dim", width=12, anchor="w").pack(side=tk.LEFT)
        self.openai_dim_var = tk.StringVar(value=os.getenv("OPENAI_EMBEDDING_DIM", "3072"))
        self.openai_dim_combo = ttk.Combobox(row2b, textvariable=self.openai_dim_var, 
                                              values=["256", "512", "768", "1024", "1536", "3072"], state="readonly", width=8)
        self.openai_dim_combo.pack(side=tk.LEFT, padx=(4, 6))
        self.env_entries["OPENAI_EMBEDDING_DIM"] = self.openai_dim_var
        self.openai_dim_note = ttk.Label(row2b, text="3-large native: 3072; 3-small: 1536", style="Muted.TLabel")
        self.openai_dim_note.pack(side=tk.LEFT, padx=(4, 0))

        # Hint for OpenAI
        self.openai_hint = ttk.Label(embed_card, text="", style="Muted.TLabel")
        self.openai_hint.pack(anchor="w", padx=(100, 0))
        self._update_openai_hint()

        ttk.Separator(embed_card, orient="horizontal").pack(fill=tk.X, pady=6)

        # Status badge row
        row3 = ttk.Frame(embed_card)
        row3.pack(fill=tk.X, pady=4)
        ttk.Label(row3, text="Status", width=12, anchor="w").pack(side=tk.LEFT)
        self.status_badge = ttk.Label(row3, text="🟢 Ready", font=("Helvetica", 11, "bold"))
        self.status_badge.pack(side=tk.LEFT, padx=(4, 8))
        self.dim_display = ttk.Label(row3, text="—", font=("Helvetica", 10))
        self.dim_display.pack(side=tk.LEFT, padx=(4, 8))
        self.dim_note = ttk.Label(row3, text="", style="Muted.TLabel")
        self.dim_note.pack(side=tk.LEFT)

        # Index status card
        status_card = ttk.LabelFrame(embed_card, text="Index status", padding=8, style="Panel.TFrame")
        status_card.pack(fill=tk.X, pady=(6, 4))
        self.sync_status = ttk.Label(status_card, text="🔄 Checking Pinecone index...", style="Muted.TLabel")
        self.sync_status.pack(anchor="w")
        self.index_info = ttk.Label(status_card, text="", style="Muted.TLabel")
        self.index_info.pack(anchor="w")

        # Danger zone
        danger = ttk.LabelFrame(embed_card, text="Danger zone: recreate index", padding=8)
        danger.pack(fill=tk.X, pady=(8, 0))
        danger_row = ttk.Frame(danger)
        danger_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(danger_row, text="New dimension", width=12, anchor="w").pack(side=tk.LEFT)
        self.new_dim_var = tk.StringVar(value="768")
        ttk.Combobox(
            danger_row,
            textvariable=self.new_dim_var,
            values=["768", "1536", "3072", "512", "256"],
            state="readonly",
            width=10,
        ).pack(side=tk.LEFT, padx=4)
        recreate_btn = ttk.Button(danger_row, text="🔄 Recreate Index", command=self._recreate_index)
        recreate_btn.pack(side=tk.LEFT, padx=8)
        ToolTip(recreate_btn, "Delete and recreate the Pinecone index with the selected dimension")
        ttk.Label(
            danger,
            text="Deletes ALL vectors; re-sync afterward.",
            foreground="#e74c3c",
            style="Muted.TLabel",
        ).pack(anchor="w")

        # Footer save
        save_btn = ttk.Button(container, text="💾 Save to .env", style="Accent.TButton", command=self._save)
        save_btn.pack(anchor="e", pady=(8, 0))
        ToolTip(save_btn, "Persist all settings to your .env file")

        # Auto-load index status
        self.after(100, self._load_index_status)

    # ------------------------------------------------------------------
    def _refresh_status(self):
        """Refresh inline status badges for Plaud, Pinecone, and LLM."""
        # Plaud
        cid = os.getenv("PLAUD_CLIENT_ID")
        secret = os.getenv("PLAUD_CLIENT_SECRET")
        if not cid or not secret:
            self.plaud_status.config(text="Plaud: missing client id/secret", foreground="#e74c3c")
        else:
            try:
                from src.plaud_oauth import PlaudOAuthClient

                oauth = PlaudOAuthClient()
                if oauth.is_authenticated:
                    self.plaud_status.config(text="Plaud: token loaded", foreground="#27ae60")
                else:
                    self.plaud_status.config(text="Plaud: needs auth", foreground="#e67e22")
            except Exception as exc:  # pragma: no cover - UI path
                self.plaud_status.config(text=f"Plaud: {exc}", foreground="#e74c3c")

        # Pinecone
        pkey = os.getenv("PINECONE_API_KEY")
        idx = os.getenv("PINECONE_INDEX_NAME", "transcripts")
        if not pkey:
            self.pinecone_status.config(text="Pinecone: key missing", foreground="#e74c3c")
        else:
            self.pinecone_status.config(text=f"Pinecone: {idx}", foreground="#27ae60")

        # LLM / embeddings
        provider = os.getenv("AI_PROVIDER", "google")
        gem_key = os.getenv("GEMINI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
        dim = self._get_current_dimension()
        if provider == "google":
            color = "#27ae60" if gem_key else "#e74c3c"
            suffix = "key set" if gem_key else "key missing"
            self.gemini_status.config(text=f"LLM: Gemini ({dim}d, {suffix})", foreground=color)
        else:
            color = "#27ae60" if openai_key else "#e74c3c"
            suffix = "key set" if openai_key else "key missing"
            self.gemini_status.config(text=f"LLM: OpenAI ({dim}d, {suffix})", foreground=color)

        self._update_preflight()
    
    def _load_index_status(self):
        """Load and display current Pinecone index status with color-coded badge."""
        try:
            from gui.services.index_manager import get_index_manager, sync_dimensions
            from gui.services.embedding_service import get_embedding_service
            
            manager = get_index_manager()
            info = manager.get_index_info()
            embedder = get_embedding_service()
            target_dim = embedder.dimension
            
            if info.get("exists"):
                index_dim = info["dimension"]
                vectors = info["vector_count"]
                namespaces = ", ".join(info.get("namespaces", [])) or "default"
                
                self.sync_status.config(text=f"✅ Index synced: {index_dim}d vectors")
                self.index_info.config(text=f"   {vectors:,} vectors in namespaces: {namespaces}")
                self.dim_display.config(text=f"Index: {index_dim}d | Provider: {target_dim}d")
                
                # Color-coded status badge
                if index_dim == target_dim:
                    self.status_badge.config(text="🟢 Matched", foreground="#27ae60")
                    self.dim_note.config(text="Index and provider dimensions match", foreground="#27ae60")
                else:
                    self.status_badge.config(text="🟡 Mismatch", foreground="#e67e22")
                    self.dim_note.config(text=f"Index {index_dim}d ≠ provider {target_dim}d — use Auto-fix in Pinecone tab", foreground="#e67e22")
                
                # Sync embedding service to match
                dim, action = sync_dimensions()
                if action == "auto_adjusted":
                    self.dim_note.config(text="(auto-adjusted to match index)", foreground="#e67e22")
            else:
                self.sync_status.config(text="📦 No index yet - will create on first use")
                self.index_info.config(text=f"   Will create at {target_dim}d to match provider")
                self.dim_display.config(text=f"Provider: {target_dim}d")
                self.status_badge.config(text="🟢 Ready", foreground="#27ae60")
                self.dim_note.config(text="New index will match provider dimension", foreground="#27ae60")
                
        except Exception as e:
            self.sync_status.config(text=f"⚠️ Could not check index: {e}")
    
    def _recreate_index(self):
        """Recreate Pinecone index with new dimension (DESTRUCTIVE)."""
        new_dim = int(self.new_dim_var.get())
        
        confirm = messagebox.askyesno(
            "⚠️ Recreate Index?",
            f"This will:\n"
            f"• DELETE all existing vectors\n"
            f"• Create new index with {new_dim}d dimension\n"
            f"• Require re-syncing all transcripts\n\n"
            f"Are you sure?",
            icon="warning"
        )
        
        if not confirm:
            return
        
        try:
            from gui.services.index_manager import get_index_manager
            from gui.services.embedding_service import get_embedding_service, EmbeddingConfig
            from pinecone import Pinecone, ServerlessSpec
            
            manager = get_index_manager()
            index_name = manager.index_name
            
            # Delete existing index
            self.sync_status.config(text="🗑️ Deleting old index...")
            self.update()
            
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            existing = [idx.name for idx in pc.list_indexes()]
            
            if index_name in existing:
                pc.delete_index(index_name)
                import time
                time.sleep(2)  # Wait for deletion
            
            # Create new index
            self.sync_status.config(text=f"📦 Creating {new_dim}d index...")
            self.update()
            
            pc.create_index(
                name=index_name,
                dimension=new_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            
            # Update embedding service
            config = EmbeddingConfig(dimension=new_dim)
            get_embedding_service(config)
            
            # Clear cache and reload
            manager.clear_cache()
            
            messagebox.showinfo(
                "✅ Index Recreated",
                f"New index created with {new_dim}d dimension.\n\n"
                f"You'll need to re-sync your transcripts."
            )
            
            self._load_index_status()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to recreate index: {e}")
            self._load_index_status()

    def _save(self):
        updated = {key: var.get() for key, var in self.env_entries.items()}
        self.call('save_settings', updated)
        self._refresh_status()

    def _on_provider_change(self, event=None):
        """Warn user when switching providers if vectors exist."""
        provider = self.provider_var.get()
        dim = self._get_current_dimension()
        
        messagebox.showinfo(
            "Provider Changed",
            f"Switched to {provider}.\n\n"
            f"New embeddings will use {dim}d vectors.\n"
            f"If your Pinecone index has a different dimension, "
            f"use the Auto-fix button in the Pinecone tab or recreate the index."
        )
        self._update_hints()

    def _on_model_change(self, event=None):
        """Update hints when model changes; disable dim selector for ada-002."""
        self._update_openai_dim_state()
        self._update_hints()

    def _update_openai_dim_state(self):
        """Enable/disable OpenAI dimension selector based on model (ada-002 is fixed)."""
        model = self.openai_model_var.get()
        if model == "text-embedding-ada-002":
            # ada-002 doesn't support dimensions param - fixed at 1536
            self.openai_dim_combo.config(state="disabled")
            self.openai_dim_var.set("1536")
            self.openai_dim_note.config(text="ada-002 is fixed at 1536d (no reduction)")
        else:
            self.openai_dim_combo.config(state="readonly")
            native = "3072" if "3-large" in model else "1536"
            self.openai_dim_note.config(text=f"Native: {native}d; can reduce for cost savings")

    def _get_current_dimension(self) -> int:
        """Get the dimension that will be used for new embeddings."""
        provider = self.provider_var.get()
        if provider == "openai":
            return int(self.openai_dim_var.get())
        else:
            return int(self.gemini_dim_var.get())

    def _update_hints(self):
        """Update all hint labels based on current selections."""
        self._update_openai_hint()
        # Update dimension display based on provider
        provider = self.provider_var.get()
        dim = self._get_current_dimension()
        self.dim_display.config(text=f"{dim}d")

    def _update_openai_hint(self):
        """Update OpenAI model hint with dimension and use case."""
        model = self.openai_model_var.get()
        dim = self.openai_dim_var.get()
        
        if model == "text-embedding-3-large":
            native = 3072
            cost = "~$0.13/1M tokens"
        elif model == "text-embedding-3-small":
            native = 1536
            cost = "~$0.02/1M tokens"
        else:  # ada-002
            native = 1536
            cost = "~$0.10/1M tokens"
        
        # Note if using reduced dimensions
        if int(dim) < native and model != "text-embedding-ada-002":
            hint = f"{dim}d (reduced from {native}d) · {cost}"
            color = "#e67e22"  # orange for reduced
        else:
            hint = f"{native}d native · {cost}"
            color = "#3498db" if "3-large" in model else "#27ae60"
        
        self.openai_hint.config(text=hint, foreground=color)

    def _update_preflight(self):
        """Aggregate readiness summary for keys/tokens/index."""
        checks = []

        # Plaud keys
        cid = os.getenv("PLAUD_CLIENT_ID")
        secret = os.getenv("PLAUD_CLIENT_SECRET")
        redirect = os.getenv("PLAUD_REDIRECT_URI", "http://localhost:8080/callback")
        if cid and secret:
            checks.append("Plaud keys: ok")
        else:
            checks.append("Plaud keys: MISSING")
        checks.append(f"Redirect: {redirect}")

        # Plaud token
        try:
            from src.plaud_oauth import PlaudOAuthClient

            oauth = PlaudOAuthClient()
            if oauth.is_authenticated:
                checks.append("Plaud token: loaded")
            else:
                checks.append("Plaud token: needs auth")
        except Exception as exc:  # pragma: no cover - UI
            checks.append(f"Plaud token: {exc}")

        # Pinecone
        pkey = os.getenv("PINECONE_API_KEY")
        idx = os.getenv("PINECONE_INDEX_NAME", "transcripts")
        if pkey:
            checks.append(f"Pinecone: key set, index '{idx}'")
        else:
            checks.append("Pinecone: key MISSING (tab will be limited)")

        # LLM
        provider = os.getenv("AI_PROVIDER", "google")
        gem_key = os.getenv("GEMINI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
        dim = self._get_current_dimension()
        if provider == "google":
            checks.append(f"LLM: Gemini {dim}d ({'key set' if gem_key else 'key missing'})")
        else:
            checks.append(f"LLM: OpenAI {dim}d ({'key set' if openai_key else 'key missing'})")

        self.preflight_label.config(text="\n".join(checks))

    # ------------------------------------------------------------------
    def _validate_token(self):
        """Validate Plaud token via /users/current; auto-refresh if possible."""
        self.plaud_status.config(text="Plaud: validating...", foreground="#e67e22")

        def task():
            from src.plaud_oauth import PlaudOAuthClient
            from src.plaud_client import PlaudClient

            oauth = PlaudOAuthClient()
            client = PlaudClient(oauth)
            user = client.get_user()
            return user

        def done(result):
            if isinstance(result, Exception):
                self.plaud_status.config(text=f"Plaud: {result}", foreground="#e74c3c")
                try:
                    messagebox.showerror("Plaud", f"Token check failed: {result}")
                except Exception:
                    pass
                return
            email = result.get("email") or result.get("username") or "ok"
            self.plaud_status.config(text=f"Plaud: ok ({email})", foreground="#27ae60")
            try:
                messagebox.showinfo("Plaud", f"Token valid. User: {email}")
            except Exception:
                pass

        run_async(task, done, tk_root=self.winfo_toplevel())

    def _copy_consent_url(self):
        """Generate consent URL and copy to clipboard."""
        try:
            from src.plaud_oauth import PlaudOAuthClient

            client = PlaudOAuthClient()
            url, _ = client.get_authorization_url()
            top = self.winfo_toplevel()
            top.clipboard_clear()
            top.clipboard_append(url)
            self.plaud_status.config(text="Plaud: consent URL copied", foreground="#3498db")
            messagebox.showinfo("Plaud", "Consent URL copied to clipboard. Paste into a browser to authorize.")
        except Exception as exc:  # pragma: no cover - UI
            self.plaud_status.config(text=f"Plaud: {exc}", foreground="#e74c3c")
            try:
                messagebox.showerror("Plaud", f"Could not build consent URL: {exc}")
            except Exception:
                pass

    def _open_env(self):
        """Open the .env file in the default editor."""
        import subprocess
        from pathlib import Path

        env_path = Path(__file__).resolve().parents[2] / ".env"
        if not env_path.exists():
            try:
                env_path.touch()
            except Exception:
                messagebox.showerror(".env", "Could not create .env file")
                return
        try:
            subprocess.Popen(["open", str(env_path)])
        except Exception as exc:  # pragma: no cover
            messagebox.showerror(".env", f"Could not open .env: {exc}")

    def _restart_app(self):
        """Persist settings and restart the GUI app."""
        try:
            self._save()
        except Exception:
            pass
        try:
            import os, sys, subprocess
            from pathlib import Path
            python = sys.executable or "python"
            script = Path(sys.argv[0]).resolve()
            subprocess.Popen([python, str(script)])
            # Close current app
            self.winfo_toplevel().destroy()
        except Exception as exc:  # pragma: no cover - UI path
            messagebox.showerror("Restart", f"Could not restart: {exc}")
