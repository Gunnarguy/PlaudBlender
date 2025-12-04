import os
import tkinter as tk
from tkinter import ttk, messagebox
from gui.views.base import BaseView
from gui.state import state


class SettingsView(BaseView):
    def _build(self):
        # ================================================================
        # ENVIRONMENT VARIABLES SECTION
        # ================================================================
        frame = ttk.LabelFrame(self, text="🔐 Environment Variables", padding=8, style="Panel.TFrame")
        frame.pack(fill=tk.X, pady=(0, 10))

        self.env_entries = {}
        keys = [
            "PLAUD_CLIENT_ID",
            "PLAUD_CLIENT_SECRET",
            "PLAUD_REDIRECT_URI",
            "PINECONE_API_KEY",
            "PINECONE_INDEX_NAME",
            "GEMINI_API_KEY",
        ]

        for idx, key in enumerate(keys):
            ttk.Label(frame, text=key, style="Muted.TLabel").grid(row=idx, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=os.getenv(key, ""))
            entry = ttk.Entry(frame, textvariable=var, width=50)
            entry.grid(row=idx, column=1, padx=6, pady=2, sticky="ew")
            self.env_entries[key] = var

        frame.columnconfigure(1, weight=1)

        # ================================================================
        # SMART EMBEDDING CONFIG (Auto-syncs with Pinecone)
        # ================================================================
        embed_frame = ttk.LabelFrame(self, text="🧮 Embedding Configuration (Auto-Managed)", padding=8, style="Panel.TFrame")
        embed_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Status display - shows current sync state
        status_frame = ttk.Frame(embed_frame)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.sync_status = ttk.Label(
            status_frame, 
            text="🔄 Checking Pinecone index...",
            style="Muted.TLabel"
        )
        self.sync_status.pack(anchor="w")
        
        self.index_info = ttk.Label(
            status_frame, 
            text="",
            style="Muted.TLabel"
        )
        self.index_info.pack(anchor="w")
        
        # Model selector (doesn't affect index compatibility)
        model_row = ttk.Frame(embed_frame)
        model_row.pack(fill=tk.X, pady=2)
        
        ttk.Label(model_row, text="Model:", width=15, anchor="w").pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value="gemini-embedding-001")
        model_combo = ttk.Combobox(
            model_row,
            textvariable=self.model_var,
            values=[
                "gemini-embedding-001",  # Latest (June 2025)
                "models/text-embedding-004",  # Legacy
            ],
            state="readonly",
            width=35
        )
        model_combo.pack(side=tk.LEFT, padx=5)
        ttk.Label(model_row, text="✓ Latest recommended", style="Muted.TLabel", foreground="#27ae60").pack(side=tk.LEFT)
        
        # Dimension display (READ ONLY - determined by index)
        dim_row = ttk.Frame(embed_frame)
        dim_row.pack(fill=tk.X, pady=2)
        
        ttk.Label(dim_row, text="Dimension:", width=15, anchor="w").pack(side=tk.LEFT)
        self.dim_display = ttk.Label(dim_row, text="Auto-detected", font=("Helvetica", 11, "bold"))
        self.dim_display.pack(side=tk.LEFT, padx=5)
        self.dim_note = ttk.Label(dim_row, text="(synced with Pinecone index)", style="Muted.TLabel")
        self.dim_note.pack(side=tk.LEFT)
        
        # Info box explaining the auto-sync
        info_frame = ttk.Frame(embed_frame)
        info_frame.pack(fill=tk.X, pady=(10, 5))
        
        info_text = """ℹ️  Dimensions are AUTOMATICALLY managed:
   • If you have an existing Pinecone index → we use its dimension
   • If no index exists → we create one with 768d (recommended)
   • Model changes work instantly (no index recreation needed)
   
   This ensures you can NEVER have a dimension mismatch error."""
        
        ttk.Label(info_frame, text=info_text, style="Muted.TLabel", justify="left").pack(anchor="w")
        
        # Advanced: Force recreate index with new dimension
        adv_frame = ttk.LabelFrame(embed_frame, text="⚠️ Advanced: Change Index Dimension", padding=5)
        adv_frame.pack(fill=tk.X, pady=(10, 0))
        
        adv_row = ttk.Frame(adv_frame)
        adv_row.pack(fill=tk.X)
        
        ttk.Label(adv_row, text="New dimension:", width=15, anchor="w").pack(side=tk.LEFT)
        self.new_dim_var = tk.StringVar(value="768")
        dim_combo = ttk.Combobox(
            adv_row,
            textvariable=self.new_dim_var,
            values=["768", "1536", "3072", "512", "256"],
            state="readonly",
            width=10
        )
        dim_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            adv_row, 
            text="🔄 Recreate Index", 
            command=self._recreate_index
        ).pack(side=tk.LEFT, padx=10)
        
        ttk.Label(
            adv_frame, 
            text="⚠️ This DELETES all vectors and creates a new index!",
            foreground="#e74c3c",
            style="Muted.TLabel"
        ).pack(anchor="w", pady=(5, 0))

        # ================================================================
        # SAVE BUTTON
        # ================================================================
        ttk.Button(self, text="💾 Save Environment", style="Accent.TButton", command=self._save).pack(anchor="e", pady=(10, 0))
        
        # Auto-load index status
        self.after(100, self._load_index_status)
    
    def _load_index_status(self):
        """Load and display current Pinecone index status."""
        try:
            from gui.services.index_manager import get_index_manager, sync_dimensions
            
            manager = get_index_manager()
            info = manager.get_index_info()
            
            if info.get("exists"):
                dim = info["dimension"]
                vectors = info["vector_count"]
                namespaces = ", ".join(info.get("namespaces", [])) or "default"
                
                self.sync_status.config(text=f"✅ Index synced: {dim}d vectors")
                self.index_info.config(text=f"   {vectors:,} vectors in namespaces: {namespaces}")
                self.dim_display.config(text=f"{dim}d")
                
                # Sync embedding service to match
                dim, action = sync_dimensions()
                if action == "auto_adjusted":
                    self.dim_note.config(text="(auto-adjusted to match index)", foreground="#e67e22")
            else:
                self.sync_status.config(text="📦 No index yet - will create on first use")
                self.index_info.config(text="   Default: 768d (recommended balance)")
                self.dim_display.config(text="768d (default)")
                
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
