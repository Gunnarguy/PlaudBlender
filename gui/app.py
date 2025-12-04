import os
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable, Optional

from gui.theme import ModernTheme
from gui.state import state
from gui.utils.async_tasks import run_async
from gui.utils.logging import log
from gui.components.status_bar import StatusBar
from gui.views.dashboard import DashboardView
from gui.views.transcripts import TranscriptsView
from gui.views.pinecone import PineconeView
from gui.views.search import SearchView
from gui.views.settings import SettingsView
from gui.views.logs import LogsView
from gui.services import transcripts_service, pinecone_service, search_service, settings_service
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
            'view_details': self.view_transcript,
            'export_selected': self.export_transcripts,
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
            # Search / settings
            'perform_search': self.perform_search,
            'perform_cross_namespace_search': self.perform_cross_namespace_search,
            'search_full_text': self.search_full_text,
            'search_summaries': self.search_summaries,
            'goto_search': lambda: self.switch_view('search'),
            'goto_settings': lambda: self.switch_view('settings'),
            'sync_all': self.sync_all,
            'generate_mindmap': self.generate_mindmap,
            'refresh_indexes': self.refresh_indexes,
            'save_settings': self.save_settings,
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
            ('settings', '⚙️ Settings'),
            ('logs', '📋 Logs'),
        ]:
            btn = ttk.Button(self.sidebar, text=icon, style="Nav.TButton", command=lambda n=name: self.switch_view(n))
            btn.pack(fill=tk.X)
            self.nav_buttons[name] = btn

        self.content = ttk.Frame(container, style="Main.TFrame")
        self.content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.view_container = ttk.Frame(self.content, style="Main.TFrame")
        self.view_container.pack(fill=tk.BOTH, expand=True)

    def _create_views(self):
        self.views['dashboard'] = DashboardView(self.view_container, self.actions)
        self.views['transcripts'] = TranscriptsView(self.view_container, self.actions)
        self.views['pinecone'] = PineconeView(self.view_container, self.actions)
        self.views['search'] = SearchView(self.view_container, self.actions)
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
        self.set_status("Checking Plaud authentication", busy=True)

        def init_auth():
            client = get_oauth_client()
            return client.is_authenticated

        def after_auth(result):
            state.is_authenticated = bool(result)
            self.views['dashboard'].update_stats({'auth': state.is_authenticated})
            if state.is_authenticated:
                self.fetch_transcripts()

        self._execute_task(init_auth, after_auth)

    def set_status(self, message, busy=False):
        state.set_status(message, busy)
        self.status_bar.update_status()

    # ------------------- Transcript actions ---------------------------
    def fetch_transcripts(self):
        self.set_status("Loading transcripts", True)

        def task():
            return transcripts_service.fetch_transcripts()

        def done(result):
            self.views['transcripts'].populate(result)
            self.views['dashboard'].update_stats({'recordings': len(result)})

        self._execute_task(task, done)

    def filter_transcripts(self, query):
        filtered = transcripts_service.filter_transcripts(query)
        self.views['transcripts'].populate(filtered)

    def sync_selected(self):
        messagebox.showinfo("Sync", "Sync logic not yet wired")

    def delete_selected(self):
        messagebox.showinfo("Delete", "Delete logic not yet wired")

    def view_transcript(self):
        messagebox.showinfo("Transcript", "Display transcript viewer soon")

    def export_transcripts(self):
        messagebox.showinfo("Export", "Export logic not yet wired")

    # ------------------- Pinecone ------------------------------------
    def load_pinecone_indexes(self):
        """Fetch available indexes/namespaces and populate dropdowns, then load vectors."""
        self.set_status("Loading Pinecone indexes", True)

        def task():
            return pinecone_service.get_indexes_and_namespaces()

        def done(result):
            indexes, namespaces, current, stats = result
            view = self.views['pinecone']
            view.set_indexes(indexes, current)
            view.set_namespaces(namespaces)
            view.set_stats(stats)
            self.refresh_vectors()

        self._execute_task(task, done)

    def refresh_vectors(self):
        self.set_status("Loading Pinecone vectors", True)
        namespace = self.views['pinecone'].namespace_var.get()

        def task():
            return pinecone_service.refresh_vectors(namespace)

        def done(result):
            self.views['pinecone'].populate(result)
            self.views['dashboard'].update_stats({'pinecone': len(result)})

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
            self.refresh_vectors()

        self._execute_task(task, done)

    def refresh_indexes(self):
        client = get_pinecone_client()
        info = client.get_index_info()
        log('INFO', f"Index info: {info}")

    def select_namespace(self, namespace):
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
                values = [float(x.strip()) for x in manual_embed.split(",") if x.strip()]

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
            client = get_pinecone_client()
            vec_id = params.get("id")
            new_meta = params.get("metadata", {})
            client.index.update(id=vec_id, set_metadata=new_meta, namespace=namespace)
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
            client = get_pinecone_client()
            if params.get("delete_all"):
                client.index.delete(delete_all=True, namespace=namespace)
                return "all"
            elif params.get("filter"):
                client.index.delete(filter=params["filter"], namespace=namespace)
                return "filtered"
            else:
                ids = params.get("ids", [])
                client.index.delete(ids=ids, namespace=namespace)
                return len(ids)

        def done(result):
            log('INFO', f"Deleted vectors: {result}")
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
            for ns, count in namespaces:
                txt.insert(tk.END, f"{ns:<20} | {count:,}\n")

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

    # ------------------- Settings ------------------------------------
    def save_settings(self, values):
        settings_service.save_settings(values)
        messagebox.showinfo("Settings", "Saved. Restart app to reload keys.")

    # ------------------- Misc ----------------------------------------
    def sync_all(self):
        messagebox.showinfo("Sync", "Sync pipeline not yet wired")

    def generate_mindmap(self):
        messagebox.showinfo("Mind Map", "Mind map generator coming soon")

    def run(self):
        self.root.mainloop()

    # ------------------- Internals -----------------------------------
    def _execute_task(self, task: Callable, on_success: Optional[Callable] = None, ready_message: str = "Ready"):
        """Run *task* in the background and funnel results through a single UI-safe callback."""
        def callback(result):
            if isinstance(result, Exception):
                log('ERROR', str(result))
                self.set_status("Error")
                messagebox.showerror("PlaudBlender", str(result))
                return
            self.set_status(ready_message)
            if on_success:
                on_success(result)

        run_async(task, callback, tk_root=self.root)


def main():
    app = PlaudBlenderApp()
    app.run()
