import os
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from typing import Callable, Optional

from gui.theme import ModernTheme
from gui.state import state
from gui.utils.async_tasks import run_async
from gui.utils.tooltips import ToolTip
from gui.utils.logging import log
from gui.components.status_bar import StatusBar
from gui.views.dashboard import DashboardView
from gui.views.transcripts import TranscriptsView
from gui.views.pinecone import PineconeView
from gui.views.search import SearchView
from gui.views.settings import SettingsView
from gui.views.logs import LogsView
from gui.views.chat import ChatView
from gui.services import transcripts_service, pinecone_service, search_service, settings_service, chat_service
from gui.services.clients import get_oauth_client, get_pinecone_client
from gui.services.embedding_service import get_embedding_service, EmbeddingError


class PlaudBlenderApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PlaudBlender")
        self.root.geometry("1400x900")
        ModernTheme.apply(self.root)

        self.actions = {
            'fetch_transcripts': self.fetch_transcripts,
            'sync_selected': self.sync_selected,
            'delete_selected': self.delete_selected,
            'view_transcript': self.view_transcript,
            'view_details': self.view_details,
            'export_selected': self.export_transcripts,
            'copy_metadata': self.copy_metadata,
            'filter_transcripts': self.filter_transcripts,
            # Pinecone - Core
            'load_pinecone_indexes': self.load_pinecone_indexes,
            'refresh_vectors': self.refresh_vectors,
            'change_index': self.change_index,
            'select_namespace': self.select_namespace,
            # Pinecone - Query/Search (full SDK)
            'similarity_search': self.similarity_search,
            'fetch_by_metadata': self.fetch_by_metadata,
            'query_all_namespaces': self.query_all_namespaces,
            'fetch_by_ids': self.fetch_by_ids,
            # Pinecone - Mutations (full SDK)
            'upsert_vector': self.upsert_vector,
            'update_metadata': self.update_metadata,
            'delete_vectors': self.delete_vectors,
            # Pinecone - Namespace management
            'create_namespace': self.create_namespace,
            'delete_namespace': self.delete_namespace,
            'list_namespaces_detail': self.list_namespaces_detail,
            # Pinecone - Bulk
            'bulk_import': self.bulk_import,
            'auto_fix_dim': self.auto_fix_dim,
            # Search / settings
            'perform_search': self.perform_search,
            'perform_cross_namespace_search': self.perform_cross_namespace_search,
            'perform_rerank_search': self.perform_rerank_search,
            'search_full_text': self.search_full_text,
            'search_summaries': self.search_summaries,
            'save_search': self.save_search,
            'load_saved_search': self.load_saved_search,
            'goto_search': lambda: self.switch_view('search'),
            'goto_settings': lambda: self.switch_view('settings'),
            'sync_all': self.sync_all,
            'generate_mindmap': self.generate_mindmap,
            'refresh_indexes': self.refresh_indexes,
            'save_settings': self.save_settings,
            'reembed_all': self.reembed_all,
            'show_db_browser': self.show_db_browser,
            'chat_send': self.chat_send,
            'goto_chat': lambda: self.switch_view('chat'),
        }

        self._build_layout()
        self.status_bar = StatusBar(self.content)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.views = {}
        self._create_views()
        self.switch_view('dashboard')

        self.root.after(200, self.bootstrap)

    # ------------------------------------------------------------------
    def _build_layout(self):
        container = ttk.Frame(self.root, style="Main.TFrame")
        container.pack(fill=tk.BOTH, expand=True)

        self.sidebar = ttk.Frame(container, style="Sidebar.TFrame", width=160)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        ttk.Label(self.sidebar, text="PlaudBlender", style="Header.TLabel").pack(pady=(12, 8), padx=10, anchor='w')

        self.nav_buttons = {}
        for name, icon in [
            ('dashboard', '📊 Dashboard'),
            ('transcripts', '📝 Transcripts'),
            ('pinecone', '🌲 Pinecone'),
            ('search', '🔍 Search'),
            ('chat', '💬 Chat'),
            ('settings', '⚙️ Settings'),
            ('logs', '📋 Logs'),
        ]:
            btn = ttk.Button(self.sidebar, text=icon, style="Nav.TButton", command=lambda n=name: self.switch_view(n))
            btn.pack(fill=tk.X)
            self.nav_buttons[name] = btn

        self.content = ttk.Frame(container, style="Main.TFrame")
        self.content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Global command bar for the most common actions
        self.topbar = ttk.Frame(self.content, style="Panel.TFrame", padding=8)
        self.topbar.pack(fill=tk.X)
        ttk.Label(self.topbar, text="Quick actions", style="Muted.TLabel").pack(side=tk.LEFT, padx=(0, 8))
        btn_fetch = ttk.Button(self.topbar, text="↻ Fetch", command=self.fetch_transcripts)
        btn_fetch.pack(side=tk.LEFT, padx=2)
        ToolTip(btn_fetch, "Pull recordings from Plaud into the local database")

        btn_sync_all = ttk.Button(self.topbar, text="🔄 Sync all", style="Accent.TButton", command=self.sync_all)
        btn_sync_all.pack(side=tk.LEFT, padx=2)
        ToolTip(btn_sync_all, "Chunk, embed, and upsert all pending recordings")

        btn_search = ttk.Button(self.topbar, text="🔍 Search", command=lambda: self.switch_view('search'))
        btn_search.pack(side=tk.LEFT, padx=2)
        ToolTip(btn_search, "Open Semantic Search")

        btn_chat = ttk.Button(self.topbar, text="💬 Chat", command=lambda: self.switch_view('chat'))
        btn_chat.pack(side=tk.LEFT, padx=2)
        ToolTip(btn_chat, "Open Chat (OpenAI Responses)")

        btn_pine = ttk.Button(self.topbar, text="🌲 Pinecone", command=lambda: (self.switch_view('pinecone'), self.load_pinecone_indexes()))
        btn_pine.pack(side=tk.LEFT, padx=2)
        ToolTip(btn_pine, "Inspect vectors, namespaces, and run queries")

        self.view_container = ttk.Frame(self.content, style="Main.TFrame")
        self.view_container.pack(fill=tk.BOTH, expand=True)

    def _create_views(self):
        self.views['dashboard'] = DashboardView(self.view_container, self.actions)
        self.views['transcripts'] = TranscriptsView(self.view_container, self.actions)
        self.views['pinecone'] = PineconeView(self.view_container, self.actions)
        self.views['search'] = SearchView(self.view_container, self.actions)
        self.views['chat'] = ChatView(self.view_container, self.actions)
        self.views['settings'] = SettingsView(self.view_container, self.actions)
        self.views['logs'] = LogsView(self.view_container, self.actions)

    def switch_view(self, name):
        for view in self.views.values():
            view.pack_forget()
        view = self.views[name]
        view.pack(fill=tk.BOTH, expand=True)
        state.set_status(f"Viewing {name.title()}")
        self.status_bar.update_status()
        if hasattr(view, 'on_show'):
            view.on_show()
        for nav_name, btn in self.nav_buttons.items():
            style = "Accent.TButton" if nav_name == name else "Nav.TButton"
            btn.configure(style=style)

    # ------------------------------------------------------------------
    def bootstrap(self):
        # Lightweight env sanity check before hitting networked auth
        if self._env_preflight():
            return
        self.set_status("Checking Plaud authentication", busy=True)

        def init_auth():
            client = get_oauth_client()
            return client.is_authenticated

        def after_auth(result):
            state.is_authenticated = bool(result)
            self.views['dashboard'].update_stats({'auth': state.is_authenticated})
            if state.is_authenticated:
                self.fetch_transcripts()
                # Auto-load Pinecone indexes so dashboard selectors are ready on launch
                self.load_pinecone_indexes()

        self._execute_task(init_auth, after_auth)

    def _env_preflight(self):
        """Check critical env vars and surface missing ones in status/logs."""
        required = {
            "PLAUD_CLIENT_ID": os.getenv("PLAUD_CLIENT_ID"),
            "PLAUD_CLIENT_SECRET": os.getenv("PLAUD_CLIENT_SECRET"),
        }
        redirect = os.getenv("PLAUD_REDIRECT_URI")
        pinecone_key = os.getenv("PINECONE_API_KEY")

        missing = [k for k, v in required.items() if not v]
        warnings = []
        if not redirect:
            warnings.append("PLAUD_REDIRECT_URI (using default http://localhost:8080/callback)")
        if not pinecone_key:
            warnings.append("PINECONE_API_KEY (Pinecone features disabled)")

        if missing:
            msg = f"Missing required env: {', '.join(missing)}. Update .env and restart."
            self.set_status(msg)
            try:
                messagebox.showwarning("Configuration", msg)
            except Exception:
                pass
            return True

        if warnings:
            msg = f"Env warnings: {', '.join(warnings)}"
            self.set_status(msg)
            # Not blocking; proceed with auth
        return False

    def set_status(self, message, busy=False):
        state.set_status(message, busy)
        # Record lightweight activity for the dashboard feed
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        state.logs.append(f"[{timestamp}] STATUS: {message}")
        self.status_bar.update_status()
        # Surface to dashboard activity feed for transparency
        if 'dashboard' in self.views:
            lines = state.logs[-10:]
            self.views['dashboard'].update_activity(lines)

    # ------------------- Transcript actions ---------------------------
    def fetch_transcripts(self):
        self.set_status("Loading transcripts", True)

        def task():
            # Ingest from Plaud into SQLite, then read from DB for display
            return transcripts_service.fetch_transcripts()

        def done(result):
            self.views['transcripts'].populate(result)
            status_counts = {'raw': 0, 'processed': 0, 'indexed': 0}
            for rec in result:
                status_counts[rec.get('status', 'raw')] = status_counts.get(rec.get('status', 'raw'), 0) + 1
            self.views['dashboard'].update_stats({
                'recordings': len(result),
                'last_sync': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'status_raw': status_counts.get('raw', 0),
                'status_processed': status_counts.get('processed', 0),
                'status_indexed': status_counts.get('indexed', 0),
            })
            self.views['dashboard'].update_recent_transcripts(result)
            if not result:
                messagebox.showwarning("Transcripts", "No transcripts loaded. Check Plaud auth or network.")
            self.set_status("Ready")

        self._execute_task(task, done)

    def filter_transcripts(self, query):
        filtered = transcripts_service.filter_transcripts(query)
        self.views['transcripts'].populate(filtered)

    def sync_selected(self):
        rec = self._get_selected_transcript()
        if not rec:
            return
        self.set_status("Syncing transcript", True)

        def task():
            rec_id = str(rec.get('id'))
            return transcripts_service.sync_recording(rec_id)

        def done(_):
            messagebox.showinfo("Sync", "Transcript processed, embedded, and upserted to Pinecone")
            self.refresh_vectors()

        self._execute_task(task, done)

    def delete_selected(self):
        rec = self._get_selected_transcript()
        if not rec:
            return
        rec_id = str(rec.get('id'))
        if not messagebox.askyesno("Delete", f"Delete vector(s) for recording {rec_id} in Pinecone?"):
            return

        def task():
            from gui.services import pinecone_service
            # Delete from both primary namespaces
            res_full = pinecone_service.delete_vectors(ids=[rec_id], namespace="full_text")
            res_sum = pinecone_service.delete_vectors(ids=[rec_id], namespace="summaries")
            return res_full, res_sum

        def done(res):
            messagebox.showinfo("Delete", f"Delete results: full_text={res[0]}, summaries={res[1]}")
            self.refresh_vectors()

        self._execute_task(task, done)

    def view_transcript(self):
        rec = self._get_selected_transcript()
        if not rec:
            return
        self._show_metadata_dialog(rec)

    def view_details(self):
        rec = self._get_selected_transcript()
        if not rec:
            return

        self.set_status("Loading transcript", True)

        def task():
            return transcripts_service.get_transcript_text(str(rec.get('id')))

        def done(text):
            self._show_transcript_dialog(rec, text)

        self._execute_task(task, done)

    def export_transcripts(self):
        rec = self._get_selected_transcript()
        if not rec:
            return
        import json
        from tkinter import filedialog

        rec_id = str(rec.get('id'))
        self.set_status("Exporting transcript", True)

        def task():
            text = transcripts_service.get_transcript_text(rec_id)
            return {"id": rec_id, "metadata": rec, "text": text}

        def done(payload):
            path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
            if path:
                with open(path, "w") as f:
                    json.dump(payload, f, indent=2, default=str)
                messagebox.showinfo("Export", f"Saved to {path}")
            self.set_status("Ready")

        self._execute_task(task, done)

    def copy_metadata(self):
        # Delegate to the view helper to copy metadata and Plaud extras
        if 'transcripts' in self.views:
            self.views['transcripts'].copy_metadata()

    # ------------------- Pinecone ------------------------------------
    def load_pinecone_indexes(self):
        """Fetch available indexes/namespaces and populate dropdowns, then load vectors."""
        self.set_status("Loading Pinecone indexes", True)

        def task():
            return pinecone_service.get_indexes_and_namespaces()

        def done(result):
            indexes, namespaces, current, stats = result

            # If dimension mismatch, try to create/switch to a matching index
            if stats.get('dim_mismatch'):
                target_dim = stats.get('target_dim') or stats.get('dimension')
                base = current or os.getenv('PINECONE_INDEX_NAME', 'transcripts')
                candidate = pinecone_service.find_matching_index(target_dim) or f"{base}-{target_dim}"
                log('WARNING', f"Dimension mismatch: index dim={stats.get('dimension')} vs target={target_dim}; "
                               f"auto-switch/create -> {candidate}")
                try:
                    pinecone_service.ensure_matching_index(candidate, target_dim)
                    indexes, namespaces, current, stats = pinecone_service.get_indexes_and_namespaces()
                    messagebox.showinfo(
                        "Pinecone",
                        f"Auto-switched to '{candidate}' at {target_dim}d to match provider",
                    )
                except Exception as e:
                    messagebox.showwarning(
                        "Pinecone",
                        "Dimension mismatch: index dim does not match your embedding provider/model.\n\n"
                        f"Tried to auto-create/switch to '{candidate}' ({target_dim}d) and failed: {e}\n\n"
                        "How to fix: either switch to an index with the same dimension as your provider, "
                        "or pick a provider/model that outputs the index dimension, then re-embed your data.",
                    )

            view = self.views['pinecone']
            view.set_indexes(indexes, current)
            view.set_namespaces(namespaces)
            view.set_stats(stats)
            # Keep dashboard selectors in sync
            self.views['dashboard'].set_indexes(indexes, current)
            self.views['dashboard'].set_namespaces(namespaces, namespaces[0] if namespaces else None)
            self.views['dashboard'].update_stats({
                'pinecone': stats.get('vectors', 0),
                'pinecone_namespaces': stats.get('namespaces', 0),
                'pinecone_dim': stats.get('dimension', '—'),
                'pinecone_metric': stats.get('metric', '—'),
                'pinecone_dim_mismatch': stats.get('dim_mismatch', False),
                'pinecone_index': stats.get('index_name', current),
                'pinecone_provider': stats.get('provider', 'google'),
                'pinecone_namespace': view.namespace_var.get() if hasattr(view, 'namespace_var') else None,
            })
            self.refresh_vectors()

        self._execute_task(task, done)

    def refresh_vectors(self):
        self.set_status("Loading Pinecone vectors", True)
        namespace = self.views['pinecone'].namespace_var.get()

        def task():
            return pinecone_service.refresh_vectors(namespace)

        def done(result):
            self.views['pinecone'].populate(result)
            self.views['dashboard'].update_stats({'pinecone': len(result), 'last_pinecone': datetime.now().strftime('%Y-%m-%d %H:%M')})

        self._execute_task(task, done)

    def auto_fix_dim(self):
        """One-click: align index dimension to current provider/model and optionally re-embed."""
        self.set_status("Auto-fixing dimension", True)

        def task():
            # Get fresh stats
            indexes, namespaces, current, stats = pinecone_service.get_indexes_and_namespaces()
            if not stats.get('dim_mismatch'):
                return {"status": "ok", "message": "No mismatch", "stats": stats, "current": current, "indexes": indexes, "namespaces": namespaces}

            target_dim = stats.get('target_dim') or stats.get('dimension')
            base = current or os.getenv('PINECONE_INDEX_NAME', 'transcripts')
            candidate = pinecone_service.find_matching_index(target_dim) or f"{base}-{target_dim}"

            pinecone_service.ensure_matching_index(candidate, target_dim)
            # Refresh after switch
            indexes2, namespaces2, current2, stats2 = pinecone_service.get_indexes_and_namespaces()
            return {
                "status": "switched",
                "candidate": candidate,
                "target_dim": target_dim,
                "indexes": indexes2,
                "namespaces": namespaces2,
                "current": current2,
                "stats": stats2,
            }

        def done(res):
            self.set_status("Ready")
            if res.get("status") == "ok":
                messagebox.showinfo("Pinecone", "No dimension mismatch detected.")
                return

            candidate = res.get("candidate")
            target_dim = res.get("target_dim")
            indexes = res.get("indexes")
            namespaces = res.get("namespaces")
            current = res.get("current")
            stats = res.get("stats")

            # Update view with new index/namespace/stats
            view = self.views['pinecone']
            view.set_indexes(indexes, current)
            view.set_namespaces(namespaces)
            view.set_stats(stats)
            self.views['dashboard'].update_stats({
                'pinecone': stats.get('vectors', 0),
                'pinecone_namespaces': stats.get('namespaces', 0),
                'pinecone_dim': stats.get('dimension', '—'),
                'pinecone_metric': stats.get('metric', '—'),
                'pinecone_dim_mismatch': stats.get('dim_mismatch', False),
                'pinecone_index': stats.get('index_name', current),
                'pinecone_provider': stats.get('provider', 'google'),
            })

            # Ask to re-embed
            if messagebox.askyesno(
                "Pinecone",
                f"Switched to '{candidate}' at {target_dim}d to match your embedding provider/model.\n\n"
                "Re-embed all recordings into this index now?",
            ):
                self.reembed_all()
            else:
                messagebox.showinfo(
                    "Pinecone",
                    f"Using '{candidate}' ({target_dim}d). Re-embed later via Pinecone > Re-embed all.",
                )

        self._execute_task(task, done)

    def change_index(self, index_name):
        self.set_status("Switching index", True)

        def task():
            return pinecone_service.switch_index(index_name)

        def done(result):
            namespaces, stats = result
            view = self.views['pinecone']
            view.set_namespaces(namespaces)
            view.set_stats(stats)
            view.index_var.set(index_name)
            # Sync dashboard selectors
            self.views['dashboard'].set_indexes(view.index_dropdown['values'], index_name)
            self.views['dashboard'].set_namespaces(namespaces, namespaces[0] if namespaces else None)
            self.views['dashboard'].update_stats({
                'pinecone': stats.get('vectors', 0),
                'pinecone_namespaces': stats.get('namespaces', 0),
                'pinecone_dim': stats.get('dimension', '—'),
                'pinecone_metric': stats.get('metric', '—'),
                'pinecone_dim_mismatch': stats.get('dim_mismatch', False),
                'pinecone_index': stats.get('index_name', index_name),
                'pinecone_provider': stats.get('provider', 'google'),
                'pinecone_namespace': namespaces[0] if namespaces else None,
            })
            self.refresh_vectors()

        self._execute_task(task, done)

    def reembed_all(self):
        target_index = self.views['pinecone'].index_var.get() or get_pinecone_client().index_name
        target_namespace = self.views['pinecone'].namespace_var.get() or "full_text"
        embedder = get_embedding_service()
        self.set_status("Re-embedding all recordings", True)

        def task():
            return pinecone_service.reembed_all_into_index(target_index, target_namespace)

        def done(result):
            if isinstance(result, dict):
                msg = (
                    f"Upserted {result.get('upserted', 0)} of {result.get('total', 0)} recordings "
                    f"into {result.get('index')}/{result.get('namespace')}\n"
                    f"Failed: {result.get('failed', 0)}"
                )
                messagebox.showinfo("Re-embed", msg)
            else:
                messagebox.showinfo("Re-embed", "Re-embedding completed")
            self.refresh_vectors()

        self._execute_task(task, done)

    def refresh_indexes(self):
        client = get_pinecone_client()
        info = client.get_index_info()
        log('INFO', f"Index info: {info}")

    def select_namespace(self, namespace):
        # Keep both views aligned
        if 'pinecone' in self.views:
            try:
                self.views['pinecone'].namespace_var.set(namespace)
            except Exception:
                pass
        if 'dashboard' in self.views:
            try:
                self.views['dashboard'].namespace_var.set(namespace)
            except Exception:
                pass
            # Update snapshot label immediately
            self.views['dashboard'].update_pinecone_snapshot({'namespace': namespace})
        self.refresh_vectors()

    # ─────────────────────────────────────────────────────────────────
    # Pinecone Full SDK Methods
    # ─────────────────────────────────────────────────────────────────

    def similarity_search(self, params: dict):
        """
        Full similarity search using SDK query() with all options.
        
        Uses centralized EmbeddingService for consistent embeddings.
        """
        self.set_status("Searching", True)

        def task():
            client = get_pinecone_client()
            method = params.get("method", "text")
            query_input = params.get("query", "")
            top_k = params.get("top_k", 10)
            namespace = params.get("namespace")
            include_meta = params.get("include_metadata", True)
            include_vals = params.get("include_values", False)
            flt = params.get("filter")

            query_params = {
                "top_k": top_k,
                "namespace": namespace,
                "include_metadata": include_meta,
                "include_values": include_vals,
            }
            if flt:
                query_params["filter"] = flt

            if method == "text":
                # Use centralized embedding service
                embedding_service = get_embedding_service()
                query_params["vector"] = embedding_service.embed_query(query_input)
                log('INFO', f"Generated {len(query_params['vector'])}-dim query embedding")
            elif method == "id":
                query_params["id"] = query_input
            else:  # raw vector
                query_params["vector"] = [float(x.strip()) for x in query_input.split(",") if x.strip()]

            results = client.index.query(**query_params)
            return [{"id": m.id, "metadata": getattr(m, 'metadata', {}), "score": getattr(m, 'score', 0)} for m in results.matches]

        def done(matches):
            from gui.services.pinecone_service import _format_vector
            formatted = [_format_vector(type('V', (), {"id": m["id"], "metadata": m["metadata"]})()) for m in matches]
            self.views['pinecone'].populate(formatted)
            log('INFO', f"Found {len(matches)} matches")

        self._execute_task(task, done)

    def fetch_by_metadata(self, params: dict):
        """Use SDK fetch_by_metadata() - no embedding required."""
        self.set_status("Filtering by metadata", True)
        namespace = self.views['pinecone'].namespace_var.get() or None

        def task():
            client = get_pinecone_client()
            flt = params.get("filter", {})
            limit = params.get("limit", 100)

            all_vectors = []
            pagination_token = None

            # Paginate through results
            for _ in range(20):  # safety limit
                fetch_params = {"filter": flt, "namespace": namespace, "limit": min(limit, 100)}
                if pagination_token:
                    fetch_params["pagination_token"] = pagination_token

                try:
                    result = client.index.fetch_by_metadata(**fetch_params)
                    if hasattr(result, 'vectors') and result.vectors:
                        all_vectors.extend(result.vectors.values())
                    if hasattr(result, 'pagination') and result.pagination and hasattr(result.pagination, 'next'):
                        pagination_token = result.pagination.next
                    else:
                        break
                    if len(all_vectors) >= limit:
                        break
                except AttributeError:
                    # Fallback for older SDK - use zero vector query
                    stats = client.index.describe_index_stats()
                    dim = stats.dimension
                    results = client.index.query(vector=[0.0] * dim, filter=flt, top_k=limit,
                                                 include_metadata=True, namespace=namespace)
                    return results.matches
            return all_vectors

        def done(vectors):
            from gui.services.pinecone_service import _format_vector
            formatted = [_format_vector(v) for v in vectors]
            self.views['pinecone'].populate(formatted)
            log('INFO', f"Found {len(vectors)} vectors by metadata")

        self._execute_task(task, done)

    def query_all_namespaces(self, params: dict):
        """
        Use SDK query_namespaces() - parallel search across all namespaces.
        
        Uses centralized EmbeddingService for consistent embeddings.
        """
        self.set_status("Querying all namespaces", True)

        def task():
            client = get_pinecone_client()
            query_text = params.get("query", "")
            top_k = params.get("top_k", 10)
            flt = params.get("filter")

            # Get all namespaces
            namespaces = list(client.index.list_namespaces())
            ns_names = [ns.name if hasattr(ns, 'name') else str(ns) for ns in namespaces] or [""]
            if not ns_names or ns_names == [""]:
                raise RuntimeError("No namespaces available yet. Create or import vectors, then try again.")

            # Use centralized embedding service
            embedding_service = get_embedding_service()
            vector = embedding_service.embed_query(query_text)
            log('INFO', f"Generated {len(vector)}-dim embedding for cross-namespace query")

            # Get metric from index info
            info = client.pc.describe_index(client.index_name)
            metric = getattr(info, 'metric', 'cosine')

            # Query all namespaces
            results = client.index.query_namespaces(
                vector=vector,
                namespaces=ns_names,
                metric=metric,
                top_k=top_k,
                filter=flt,
                include_metadata=True
            )
            return results

        def done(results):
            from gui.services.pinecone_service import _format_vector
            all_matches = []
            if hasattr(results, 'matches'):
                for m in results.matches:
                    all_matches.append({"id": m.id, "metadata": getattr(m, 'metadata', {}), "namespace": getattr(m, 'namespace', '')})
            formatted = [_format_vector(type('V', (), {"id": m["id"], "metadata": m["metadata"]})()) for m in all_matches]
            self.views['pinecone'].populate(formatted)
            log('INFO', f"Found {len(all_matches)} matches across namespaces")

        self._execute_task(task, done)

    def fetch_by_ids(self, ids: list):
        """Fetch specific vectors by ID using SDK fetch()."""
        self.set_status("Fetching vectors", True)
        namespace = self.views['pinecone'].namespace_var.get() or None

        def task():
            client = get_pinecone_client()
            result = client.index.fetch(ids=ids, namespace=namespace)
            return list(result.vectors.values()) if result.vectors else []

        def done(vectors):
            from gui.services.pinecone_service import _format_vector
            formatted = [_format_vector(v) for v in vectors]
            self.views['pinecone'].populate(formatted)
            log('INFO', f"Fetched {len(vectors)} vectors")
            messagebox.showinfo("Fetch", f"Fetched {len(vectors)} vector(s)")

        self._execute_task(task, done)

    def upsert_vector(self, params: dict):
        """Upsert a vector with auto-embedding using centralized EmbeddingService."""
        self.set_status("Upserting vector", True)

        def task():
            client = get_pinecone_client()
            vec_id = params.get("id")
            metadata = params.get("metadata", {})
            embed_mode = params.get("embed_mode", "auto")
            manual_embed = params.get("manual_embedding")
            namespace = params.get("namespace")

            if embed_mode == "auto":
                # Get text content from metadata
                content = metadata.get("text", metadata.get("content", metadata.get("title", "")))
                if not content:
                    raise ValueError("No 'text' field in metadata for auto-embedding")

                # Use centralized embedding service
                embedding_service = get_embedding_service()
                values = embedding_service.embed_document(content)
                log('INFO', f"Generated {len(values)}-dim embedding for upsert")
            else:
                if isinstance(manual_embed, list):
                    values = manual_embed
                else:
                    values = [float(x.strip()) for x in str(manual_embed).split(",") if x.strip()]

            client.index.upsert(vectors=[(vec_id, values, metadata)], namespace=namespace)
            return vec_id

        def done(vec_id):
            log('INFO', f"Upserted vector: {vec_id}")
            messagebox.showinfo("Success", f"Vector upserted: {vec_id}")
            self.refresh_vectors()

        self._execute_task(task, done)

    def update_metadata(self, params: dict):
        """Update vector metadata using SDK update()."""
        self.set_status("Updating metadata", True)
        namespace = self.views['pinecone'].namespace_var.get() or None

        def task():
            from gui.services import pinecone_service
            vec_id = params.get("id")
            new_meta = params.get("metadata", {})
            ok = pinecone_service.update_vector_metadata(vec_id, new_meta, namespace=namespace or "")
            if not ok:
                raise RuntimeError("Failed to update metadata")
            return vec_id

        def done(vec_id):
            log('INFO', f"Updated metadata for: {vec_id}")
            messagebox.showinfo("Success", f"Metadata updated: {vec_id}")
            self.refresh_vectors()

        self._execute_task(task, done)

    def delete_vectors(self, params: dict):
        """Delete vectors using SDK delete() with multiple modes."""
        self.set_status("Deleting vectors", True)
        namespace = self.views['pinecone'].namespace_var.get() or None

        def task():
            from gui.services import pinecone_service
            return pinecone_service.delete_vectors(
                ids=params.get("ids"),
                flt=params.get("filter"),
                delete_all=params.get("delete_all", False),
                namespace=namespace or "",
            )

        def done(result):
            log('INFO', f"Deleted vectors: {result}")
            messagebox.showinfo("Delete", f"Deleted: {result}")
            self.refresh_vectors()

        self._execute_task(task, done)

    def create_namespace(self, name: str):
        """Create namespace by upserting a placeholder vector (Pinecone creates namespaces implicitly)."""
        self.set_status("Creating namespace", True)

        def task():
            client = get_pinecone_client()
            # Pinecone SDK doesn't have create_namespace() - namespaces are created implicitly on upsert
            # Get dimension from index stats
            stats = client.index.describe_index_stats()
            dim = stats.dimension
            # Upsert a placeholder vector to create the namespace
            # Use small non-zero values (Pinecone rejects all-zero vectors)
            placeholder_id = f"__namespace_placeholder_{name}"
            placeholder_vector = [1e-7] * dim  # tiny non-zero values
            placeholder_vector[0] = 0.001  # ensure at least one meaningful value
            client.index.upsert(
                vectors=[(placeholder_id, placeholder_vector, {"_placeholder": True, "_namespace": name})],
                namespace=name
            )
            return name

        def done(ns):
            log('INFO', f"Created namespace: {ns}")
            messagebox.showinfo("Success", f"Namespace '{ns}' created")
            self.load_pinecone_indexes()

        self._execute_task(task, done)

    def delete_namespace(self, name: str):
        """Delete namespace by deleting all vectors in it (Pinecone removes empty namespaces automatically)."""
        self.set_status("Deleting namespace", True)

        def task():
            client = get_pinecone_client()
            # Pinecone SDK doesn't have delete_namespace() - delete all vectors to remove namespace
            client.index.delete(delete_all=True, namespace=name)
            return name

        def done(ns):
            log('INFO', f"Deleted namespace: {ns}")
            messagebox.showinfo("Namespace", f"Namespace '{ns}' deleted")
            self.load_pinecone_indexes()

        self._execute_task(task, done)

    def list_namespaces_detail(self):
        """Show detailed namespace list with vector counts."""
        self.set_status("Listing namespaces", True)

        def task():
            client = get_pinecone_client()
            stats = client.index.describe_index_stats()
            namespaces = []
            if hasattr(stats, 'namespaces') and stats.namespaces:
                for ns_name, ns_stats in stats.namespaces.items():
                    count = getattr(ns_stats, 'vector_count', 0)
                    namespaces.append((ns_name or "(default)", count))
            return namespaces

        def done(namespaces):
            # Show in a dialog
            d = tk.Toplevel(self.root)
            d.title("Namespaces")
            d.geometry("400x300")
            txt = tk.Text(d, font=("JetBrains Mono", 10))
            txt.pack(fill=tk.BOTH, expand=True)
            txt.insert("1.0", "Namespace           | Vectors\n" + "-" * 35 + "\n")
            if not namespaces:
                txt.insert(tk.END, "(none yet)\n")
            for ns, count in namespaces:
                txt.insert(tk.END, f"{ns:<20} | {count:,}\n")
            ttk.Button(d, text="Close", command=d.destroy).pack(pady=6)
            # Refresh selectors/stats after viewing
            self.load_pinecone_indexes()

        self._execute_task(task, done)

    def bulk_import(self, params):
        """
        Import vectors from JSON/CSV with optional auto-embedding.
        
        Uses centralized EmbeddingService for consistent embeddings.
        """
        import json as json_mod
        import csv
        import uuid

        # Handle both old-style (path string) and new-style (dict) params
        if isinstance(params, str):
            params = {"path": params, "mode": "existing", "text_field": "text", "id_field": "id", "namespace": None}

        path = params["path"]
        mode = params.get("mode", "existing")
        text_field = params.get("text_field", "text")
        id_field = params.get("id_field", "id")
        namespace = params.get("namespace") or self.views['pinecone'].namespace_var.get() or None

        self.set_status("Importing vectors", True)

        def task():
            # Read file
            if path.endswith(".csv"):
                with open(path, 'r') as f:
                    reader = csv.DictReader(f)
                    data = list(reader)
            else:
                with open(path, 'r') as f:
                    data = json_mod.load(f)
                if not isinstance(data, list):
                    data = [data]

            client = get_pinecone_client()
            vectors = []
            skipped = 0

            if mode == "auto":
                # Use centralized embedding service for batch embedding
                embedding_service = get_embedding_service()
                
                # Collect all texts first
                texts_to_embed = []
                items_to_process = []
                
                for item in data:
                    if not isinstance(item, dict):
                        skipped += 1
                        continue
                    text = item.get(text_field, "")
                    if not text:
                        # Try common fallbacks
                        text = item.get("content", item.get("transcript", item.get("title", "")))
                    if not text:
                        skipped += 1
                        continue
                    texts_to_embed.append(str(text))
                    items_to_process.append(item)
                
                if texts_to_embed:
                    # Batch embed using centralized service
                    log('INFO', f"Generating embeddings for {len(texts_to_embed)} items...")
                    embeddings = embedding_service.embed_batch(texts_to_embed)
                    log('INFO', f"Generated {len(embeddings)} embeddings (dim={len(embeddings[0]) if embeddings else 0})")
                    
                    for item, embedding in zip(items_to_process, embeddings):
                        vec_id = item.get(id_field) or str(uuid.uuid4())
                        # Build metadata (exclude the text field if large, keep everything else)
                        metadata = {k: v for k, v in item.items() if k not in ["values", "embedding"]}
                        if text_field in metadata and len(str(metadata[text_field])) > 1000:
                            metadata[text_field] = str(metadata[text_field])[:1000] + "..."
                        vectors.append((str(vec_id), embedding, metadata))
            else:
                # Use existing embeddings
                for item in data:
                    if not isinstance(item, dict):
                        skipped += 1
                        continue
                    vec_id = item.get(id_field) or item.get("id")
                    values = item.get("values", item.get("embedding", []))
                    if not vec_id or not values:
                        skipped += 1
                        continue
                    metadata = {k: v for k, v in item.items() if k not in ["values", "embedding", "id"]}
                    vectors.append((str(vec_id), values, metadata))

            if vectors:
                # Upsert in batches
                for i in range(0, len(vectors), 100):
                    batch = vectors[i:i + 100]
                    client.index.upsert(vectors=batch, namespace=namespace)
            
            return len(vectors), skipped

        def done(result):
            count, skipped = result
            msg = f"Imported {count} vectors"
            if skipped:
                msg += f" ({skipped} skipped)"
            log('INFO', msg)
            messagebox.showinfo("Import Complete", msg)
            self.refresh_vectors()

        self._execute_task(task, done)

    # ------------------- Search --------------------------------------
    # ─────────────────────────────────────────────────────────────────
    # SEARCH ACTIONS - Ultra Granular
    # ─────────────────────────────────────────────────────────────────
    
    def perform_search(self, query, limit: int = 5):
        """
        🎯 SINGLE NAMESPACE SEARCH
        
        Searches the default namespace only.
        For specific namespace search, use the dedicated methods below.
        """
        self.set_status("🔍 Searching default namespace...", True)

        def task():
            return search_service.search_single_namespace(query, namespace="", limit=limit)

        def done(text):
            self.views['search'].show_results(text)

        self._execute_task(task, done)
    
    def perform_cross_namespace_search(self, query, limit: int = 5):
        """
        🌐 CROSS-NAMESPACE SEARCH (PARALLEL)
        
        Searches BOTH full_text AND summaries namespaces in parallel.
        Results are merged and ranked by relevance.
        """
        self.set_status("🌐 Searching all namespaces...", True)

        def task():
            return search_service.search_all_namespaces(query, limit=limit)

        def done(text):
            self.views['search'].show_results(text)

        self._execute_task(task, done)
    
    def search_full_text(self, query, limit: int = 5):
        """
        📄 SEARCH FULL TEXT NAMESPACE ONLY
        
        Searches only the 'full_text' namespace containing complete transcripts.
        Use this when looking for specific passages or detailed content.
        """
        self.set_status("📄 Searching full_text namespace...", True)

        def task():
            return search_service.search_full_text(query, limit=limit)

        def done(text):
            self.views['search'].show_results(text)

        self._execute_task(task, done)
    
    def search_summaries(self, query, limit: int = 5):
        """
        📝 SEARCH SUMMARIES NAMESPACE ONLY
        
        Searches only the 'summaries' namespace containing AI syntheses.
        Use this for high-level topic matching or thematic search.
        """
        self.set_status("📝 Searching summaries namespace...", True)

        def task():
            return search_service.search_summaries(query, limit=limit)

        def done(text):
            self.views['search'].show_results(text)

        self._execute_task(task, done)

    def perform_rerank_search(self, query, limit: int = 5, model: str = "bge-reranker-v2-m3", namespaces=None):
        """
        🏆 SEARCH + RERANK (highest quality)
        
        Two-stage search: dense vector retrieval → neural reranking.
        Uses Pinecone inference API for best relevance ordering.
        
        Args:
            query: Search query
            limit: Max results after reranking
            model: Reranker model (default: bge-reranker-v2-m3)
            namespaces: Optional list of namespaces (default: all)
        """
        self.set_status("🏆 Searching + Reranking...", True)

        def task():
            return search_service.search_with_rerank(
                query=query,
                limit=limit,
                rerank_model=model,
                namespaces=namespaces,
            )

        def done(text):
            self.views['search'].show_results(text)

        self._execute_task(task, done)

    # ------------------- Chat (OpenAI Responses) --------------------
    def chat_send(self, payload: dict, on_success=None, on_finally=None):
        """Send chat payload via OpenAI Responses and return text output."""
        self.set_status("Talking to OpenAI", True)

        def task():
            return chat_service.send_response(**payload)

        def done(result):
            if on_success:
                on_success(result)

        self._execute_task(task, done, ready_message="Ready", on_finally=on_finally)

    def save_search(self, name: str, query: str):
        from gui.services import saved_searches_service as ss
        if not name or not query:
            messagebox.showwarning("Save Search", "Provide both name and query")
            return
        ss.save_search(name, query)
        messagebox.showinfo("Save Search", f"Saved '{name}'")
        # Refresh saved list
        self.views['search'].update_saved(ss.list_saved_names())

    def load_saved_search(self, name: str):
        from gui.services import saved_searches_service as ss
        data = ss.load_saved_searches()
        if name not in data:
            messagebox.showwarning("Saved Searches", "Not found")
            return
        query = data[name]
        self.views['search'].set_query(query)
        self.perform_cross_namespace_search(query, limit=int(self.views['search'].limit_var.get()))

    # ------------------- Settings ------------------------------------
    def save_settings(self, values):
        settings_service.save_settings(values)
        messagebox.showinfo("Settings", "Saved. Restart app to reload keys.")

    # ------------------- SQLite Browser ---------------------------
    def show_db_browser(self):
        """Lightweight inspector for the local SQLite database (recordings + segments)."""
        from sqlalchemy import func
        from tkinter import filedialog
        import json
        import csv
        from src.database.engine import SessionLocal, DB_PATH, init_db
        from src.database.models import Recording, Segment

        init_db()
        session = SessionLocal()
        try:
            rec_count = session.query(func.count(Recording.id)).scalar() or 0
            seg_count = session.query(func.count(Segment.id)).scalar() or 0
            size_bytes = 0
            try:
                size_bytes = os.path.getsize(DB_PATH)
            except OSError:
                pass

            # Pre-compute segment counts per recording for display
            seg_by_rec = dict(session.query(Segment.recording_id, func.count(Segment.id)).group_by(Segment.recording_id).all())

            recordings = (
                session.query(Recording)
                .order_by(Recording.created_at.desc())
                .limit(50)
                .all()
            )
            segments = (
                session.query(Segment)
                .order_by(Segment.created_at.desc())
                .limit(50)
                .all()
            )
        finally:
            session.close()

        def fmt_bytes(num):
            for unit in ["B", "KB", "MB", "GB"]:
                if num < 1024.0:
                    return f"{num:3.1f} {unit}"
                num /= 1024.0
            return f"{num:.1f} TB"

        win = tk.Toplevel(self.root)
        win.title("SQLite Browser")
        win.geometry("1000x720")

        header = ttk.Frame(win, padding=10)
        header.pack(fill=tk.X)
        ttk.Label(header, text=f"Database: {DB_PATH}", style="Muted.TLabel").pack(anchor="w")
        ttk.Label(header, text=f"Size: {fmt_bytes(size_bytes)} • Recordings: {rec_count:,} • Segments: {seg_count:,}", style="Muted.TLabel").pack(anchor="w")

        # Export / utility row
        util = ttk.Frame(win, padding=(10, 4))
        util.pack(fill=tk.X)

        def _export(table: str):
            path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv")],
                title=f"Export {table} to CSV",
            )
            if not path:
                return
            init_db()
            session = SessionLocal()
            try:
                if table == "recordings":
                    rows = session.query(Recording).order_by(Recording.created_at.desc()).all()
                    cols = ["id", "title", "duration_ms", "created_at", "status", "language", "source", "filename", "extra"]
                else:
                    rows = session.query(Segment).order_by(Segment.created_at.desc()).all()
                    cols = ["id", "recording_id", "namespace", "status", "start_ms", "end_ms", "pinecone_id", "embedding_model", "extra", "created_at", "text"]
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(cols)
                    for r in rows:
                        vals = []
                        for c in cols:
                            val = getattr(r, c, "")
                            if isinstance(val, dict):
                                val = json.dumps(val, ensure_ascii=False)
                            vals.append(val)
                        writer.writerow(vals)
                messagebox.showinfo("Export", f"Exported {len(rows)} {table} rows to {path}")
            except Exception as e:
                messagebox.showerror("Export", str(e))
            finally:
                session.close()

        ttk.Button(util, text="Export recordings CSV", command=lambda: _export("recordings")).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(util, text="Export segments CSV", command=lambda: _export("segments")).pack(side=tk.LEFT, padx=(0, 6))

        def _copy_path():
            try:
                win.clipboard_clear()
                win.clipboard_append(DB_PATH)
                messagebox.showinfo("Copied", f"DB path copied:\n{DB_PATH}")
            except Exception:
                pass

        ttk.Button(util, text="Copy DB path", command=_copy_path).pack(side=tk.LEFT, padx=(0, 6))

        # Recordings table
        rec_frame = ttk.LabelFrame(win, text="Recordings (latest 50)", padding=6, style="Panel.TLabelframe")
        rec_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 6))
        rec_cols = ("id", "title", "status", "created", "dur", "segments", "plaud_summary")
        rec_tree = ttk.Treeview(rec_frame, columns=rec_cols, show="headings", height=10, style="Treeview")
        for col, txt, w in [
            ("id", "ID", 140),
            ("title", "Title", 240),
            ("status", "Status", 80),
            ("created", "Created", 140),
            ("dur", "Duration", 80),
            ("segments", "Segments", 80),
            ("plaud_summary", "Plaud Summary", 260),
        ]:
            rec_tree.heading(col, text=txt)
            rec_tree.column(col, width=w, anchor="w")
        for rec in recordings:
            minutes = (rec.duration_ms or 0) // 60000
            seconds = ((rec.duration_ms or 0) % 60000) // 1000
            duration_str = f"{minutes}:{seconds:02d}" if rec.duration_ms else "—"
            created = rec.created_at.strftime("%Y-%m-%d %H:%M") if rec.created_at else "—"
            summary_snippet = ""
            try:
                summary_snippet = (rec.extra or {}).get("plaud_summary", "")
                if summary_snippet:
                    summary_snippet = (summary_snippet[:120] + "…") if len(summary_snippet) > 120 else summary_snippet
            except Exception:
                summary_snippet = ""

            rec_tree.insert("", tk.END, values=(
                rec.id,
                rec.title or "Untitled",
                rec.status or "raw",
                created,
                duration_str,
                seg_by_rec.get(rec.id, 0),
                summary_snippet or "",
            ))
        rec_tree.pack(fill=tk.BOTH, expand=True)
        rec_scroll = ttk.Scrollbar(rec_frame, orient=tk.VERTICAL, command=rec_tree.yview)
        rec_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        rec_tree.configure(yscrollcommand=rec_scroll.set)

        # Segments table
        seg_frame = ttk.LabelFrame(win, text="Segments (latest 50)", padding=6, style="Panel.TLabelframe")
        seg_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        seg_cols = ("id", "rec", "ns", "status", "created")
        seg_tree = ttk.Treeview(seg_frame, columns=seg_cols, show="headings", height=10, style="Treeview")
        for col, txt, w in [
            ("id", "Segment ID", 180),
            ("rec", "Recording", 140),
            ("ns", "Namespace", 90),
            ("status", "Status", 80),
            ("created", "Created", 140),
        ]:
            seg_tree.heading(col, text=txt)
            seg_tree.column(col, width=w, anchor="w")
        for seg in segments:
            created = seg.created_at.strftime("%Y-%m-%d %H:%M") if seg.created_at else "—"
            seg_tree.insert("", tk.END, values=(seg.id, seg.recording_id, seg.namespace or "full_text", seg.status or "pending", created))
        seg_tree.pack(fill=tk.BOTH, expand=True)
        seg_scroll = ttk.Scrollbar(seg_frame, orient=tk.VERTICAL, command=seg_tree.yview)
        seg_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        seg_tree.configure(yscrollcommand=seg_scroll.set)

        ttk.Label(win, text="Read-only view • shows latest 50 rows of each table", style="Muted.TLabel").pack(anchor="w", padx=10, pady=(0, 10))

    # ------------------- Misc ----------------------------------------
    def sync_all(self):
        self.set_status("Syncing all recordings", True)

        def task():
            return transcripts_service.sync_all()

        def done(summary):
            # Refresh local view from DB (no re-ingest)
            refreshed = transcripts_service.fetch_transcripts(ingest=False)
            self.views['transcripts'].populate(refreshed)
            self.views['dashboard'].update_recent_transcripts(refreshed)
            chunk = summary.get("chunk", {}) if isinstance(summary, dict) else {}
            index = summary.get("index", {}) if isinstance(summary, dict) else {}
            messagebox.showinfo(
                "Sync",
                "All recordings processed.\n\n"
                f"Chunked: {chunk.get('recordings_processed', 0)} recordings, {chunk.get('segments_created', 0)} segments\n"
                f"Indexed: {index.get('segments_processed', 0)} segments (failures: {index.get('failures', 0)})",
            )
            self.refresh_vectors()

        self._execute_task(task, done)

    def generate_mindmap(self):
        messagebox.showinfo("Mind Map", "Mind map generator coming soon")

    def run(self):
        self.root.mainloop()

    # ------------------- Internals -----------------------------------
    def _execute_task(self, task: Callable, on_success: Optional[Callable] = None, ready_message: str = "Ready", on_finally: Optional[Callable] = None):
        """Run *task* in the background and funnel results through a single UI-safe callback."""
        def callback(result):
            try:
                if isinstance(result, Exception):
                    log('ERROR', str(result))
                    self.set_status("Error")
                    messagebox.showerror("PlaudBlender", str(result))
                else:
                    self.set_status(ready_message)
                    if on_success:
                        on_success(result)
            finally:
                if on_finally:
                    on_finally()

        run_async(task, callback, tk_root=self.root)

    # ------------------- Helpers -----------------------------------
    def _get_selected_transcript(self):
        ids = self.views['transcripts'].get_selected_ids()
        if not ids:
            messagebox.showwarning("Transcripts", "Select a transcript first")
            return None
        rec_id = ids[0]
        rec = next((r for r in state.filtered_transcripts if str(r.get('id')) == str(rec_id)), None)
        if not rec:
            messagebox.showwarning("Transcripts", "Selected transcript not found")
        return rec

    def _show_transcript_dialog(self, rec: dict, transcript_text: str):
        window = tk.Toplevel(self.root)
        window.title(f"Transcript — {rec.get('display_name', 'Untitled')}")
        window.geometry("820x600")
        window.minsize(680, 480)

        wrapper = ttk.Frame(window, padding=12)
        wrapper.pack(fill=tk.BOTH, expand=True)
        wrapper.columnconfigure(0, weight=1)
        wrapper.rowconfigure(3, weight=1)

        ttk.Label(wrapper, text="Recording details", font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0, sticky="w")

        details_box = tk.Text(wrapper, height=6, wrap="word", highlightthickness=0, borderwidth=1, relief=tk.SOLID)
        details_box.insert("1.0", self._format_rec_details(rec))
        details_box.configure(state="disabled")
        details_box.grid(row=1, column=0, sticky="nsew", pady=(4, 12))

        ttk.Label(wrapper, text="Transcript", font=("TkDefaultFont", 10, "bold")).grid(row=2, column=0, sticky="w")

        transcript_box = scrolledtext.ScrolledText(wrapper, wrap="word")
        transcript_box.insert("1.0", transcript_text or "No transcript available.")
        transcript_box.configure(state="disabled")
        transcript_box.grid(row=3, column=0, sticky="nsew", pady=(4, 0))

        ttk.Button(wrapper, text="Close", command=window.destroy).grid(row=4, column=0, pady=(12, 0), sticky="e")

    def _show_metadata_dialog(self, rec: dict):
        window = tk.Toplevel(self.root)
        window.title(f"Recording — {rec.get('display_name', 'Untitled')}")
        window.geometry("520x360")
        window.minsize(420, 300)

        wrapper = ttk.Frame(window, padding=12)
        wrapper.pack(fill=tk.BOTH, expand=True)
        wrapper.columnconfigure(0, weight=1)
        wrapper.rowconfigure(1, weight=1)

        ttk.Label(wrapper, text="Recording details", font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0, sticky="w")

        details_box = tk.Text(wrapper, height=12, wrap="word", highlightthickness=0, borderwidth=1, relief=tk.SOLID)
        details_box.insert("1.0", self._format_rec_details(rec))
        details_box.configure(state="disabled")
        details_box.grid(row=1, column=0, sticky="nsew", pady=(6, 12))

        ttk.Button(wrapper, text="Close", command=window.destroy).grid(row=2, column=0, sticky="e")

    def _format_rec_details(self, rec: dict) -> str:
        fields = {
            "Title": rec.get('display_name', '—'),
            "Date": rec.get('display_date', '—'),
            "Time": rec.get('display_time', '—'),
            "Duration": rec.get('display_duration', '—'),
            "ID": rec.get('id', '—'),
            "Short ID": rec.get('short_id', '—'),
        }
        lines = [f"{k}: {v}" for k, v in fields.items()]
        return "\n".join(lines)


def main():
    app = PlaudBlenderApp()
    app.run()
