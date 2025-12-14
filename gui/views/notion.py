"""
Notion Integration View - Dynamic workspace browser and linker.

Features:
- Search your entire Notion workspace
- Browse databases and pages
- Link recordings to specific Notion pages
- Create new pages in chosen locations
- Two-way sync with visual feedback
"""
import tkinter as tk
from tkinter import ttk, messagebox
from gui.views.base import BaseView
from gui.state import state
import threading


class NotionView(BaseView):
    """Interactive Notion workspace browser and recording linker."""
    
    def _build(self):
        # Header
        header = ttk.Frame(self, style="Main.TFrame")
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(header, text="📓 Notion Integration", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            header, 
            text="Search, browse, and link recordings to your Notion workspace", 
            style="Muted.TLabel"
        ).pack(anchor="w")
        
        # Connection status bar
        status_frame = ttk.Frame(self, style="Main.TFrame")
        status_frame.pack(fill="x", pady=(0, 8))
        self.connection_label = ttk.Label(status_frame, text="⚪ Checking connection...", style="Muted.TLabel")
        self.connection_label.pack(side="left")
        ttk.Button(status_frame, text="↻ Refresh", command=self._check_connection).pack(side="right")
        
        # Main content in two columns
        content = ttk.Frame(self, style="Main.TFrame")
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)
        
        # Left panel: Notion browser
        left_panel = ttk.LabelFrame(content, text="Notion Workspace", padding=8, style="Panel.TLabelframe")
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        left_panel.rowconfigure(2, weight=1)
        left_panel.columnconfigure(0, weight=1)
        
        # Search bar
        search_frame = ttk.Frame(left_panel, style="Panel.TFrame")
        search_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        search_frame.columnconfigure(0, weight=1)
        
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        self.search_entry.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.search_entry.bind("<Return>", lambda e: self._search_notion())
        
        ttk.Button(search_frame, text="🔍 Search", command=self._search_notion).grid(row=0, column=1)
        
        # Quick filters
        filter_frame = ttk.Frame(left_panel, style="Panel.TFrame")
        filter_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        
        ttk.Button(filter_frame, text="📊 View Databases", command=lambda: self._filter_type("database")).pack(side="left", padx=2)
        ttk.Button(filter_frame, text="📄 View Pages", command=lambda: self._filter_type("page")).pack(side="left", padx=2)
        ttk.Button(filter_frame, text="📅 Recent", command=self._show_recent).pack(side="left", padx=2)
        ttk.Button(filter_frame, text="🔗 Linked", command=self._show_linked).pack(side="left", padx=2)
        
        # Results tree
        tree_frame = ttk.Frame(left_panel, style="Panel.TFrame")
        tree_frame.grid(row=2, column=0, sticky="nsew")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)
        
        cols = ("title", "type", "updated")
        self.notion_tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=15)
        self.notion_tree.heading("title", text="Title")
        self.notion_tree.heading("type", text="Type")
        self.notion_tree.heading("updated", text="Last Updated")
        self.notion_tree.column("title", width=250)
        self.notion_tree.column("type", width=80)
        self.notion_tree.column("updated", width=100)
        self.notion_tree.grid(row=0, column=0, sticky="nsew")
        
        scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.notion_tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.notion_tree.configure(yscrollcommand=scroll.set)
        
        self.notion_tree.bind("<<TreeviewSelect>>", self._on_notion_select)
        self.notion_tree.bind("<Double-1>", self._on_notion_double_click)
        
        # Right panel: Recording linker
        right_panel = ttk.LabelFrame(content, text="Link Recordings", padding=8, style="Panel.TLabelframe")
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        right_panel.rowconfigure(2, weight=1)
        right_panel.columnconfigure(0, weight=1)
        
        # Selected Notion page info
        notion_info = ttk.Frame(right_panel, style="Panel.TFrame")
        notion_info.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        
        ttk.Label(notion_info, text="Selected Notion Page:", style="Muted.TLabel").pack(anchor="w")
        self.selected_notion = ttk.Label(notion_info, text="None selected", style="TLabel")
        self.selected_notion.pack(anchor="w")
        
        # Recording selector
        rec_frame = ttk.Frame(right_panel, style="Panel.TFrame")
        rec_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        rec_frame.columnconfigure(0, weight=1)
        
        ttk.Label(rec_frame, text="Select Recording to Link:", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        
        self.recording_var = tk.StringVar()
        self.recording_combo = ttk.Combobox(rec_frame, textvariable=self.recording_var, state="readonly")
        self.recording_combo.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        
        # Recording list (unlinked)
        ttk.Label(right_panel, text="Unlinked Recordings:", style="Muted.TLabel").grid(row=2, column=0, sticky="nw")
        
        rec_tree_frame = ttk.Frame(right_panel, style="Panel.TFrame")
        rec_tree_frame.grid(row=3, column=0, sticky="nsew")
        rec_tree_frame.rowconfigure(0, weight=1)
        rec_tree_frame.columnconfigure(0, weight=1)
        
        rec_cols = ("title", "date", "status")
        self.recording_tree = ttk.Treeview(rec_tree_frame, columns=rec_cols, show="headings", height=8)
        self.recording_tree.heading("title", text="Recording")
        self.recording_tree.heading("date", text="Date")
        self.recording_tree.heading("status", text="Notion")
        self.recording_tree.column("title", width=180)
        self.recording_tree.column("date", width=80)
        self.recording_tree.column("status", width=60)
        self.recording_tree.grid(row=0, column=0, sticky="nsew")
        
        rec_scroll = ttk.Scrollbar(rec_tree_frame, orient="vertical", command=self.recording_tree.yview)
        rec_scroll.grid(row=0, column=1, sticky="ns")
        self.recording_tree.configure(yscrollcommand=rec_scroll.set)
        
        self.recording_tree.bind("<<TreeviewSelect>>", self._on_recording_select)
        
        # Action buttons
        actions = ttk.Frame(right_panel, style="Panel.TFrame")
        actions.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        
        ttk.Button(actions, text="🔗 Link Selected", command=self._link_recording).pack(side="left", padx=2)
        ttk.Button(actions, text="📤 Push to Page", command=self._push_to_page).pack(side="left", padx=2)
        ttk.Button(actions, text="➕ Create Page", command=self._create_notion_page).pack(side="left", padx=2)
        
        # Bottom: Sync controls
        sync_frame = ttk.LabelFrame(self, text="Sync Operations", padding=8, style="Panel.TLabelframe")
        sync_frame.pack(fill="x", pady=(8, 0))
        
        ttk.Button(sync_frame, text="📤 Push All to Notion", command=lambda: self.call('notion_push')).pack(side="left", padx=4)
        ttk.Button(sync_frame, text="📥 Pull from Notion", command=lambda: self.call('notion_pull')).pack(side="left", padx=4)
        ttk.Button(sync_frame, text="🔄 Full Sync", command=lambda: self.call('notion_full_sync')).pack(side="left", padx=4)
        
        self.sync_status = ttk.Label(sync_frame, text="", style="Muted.TLabel")
        self.sync_status.pack(side="right", padx=4)
        
        # Store state
        self._selected_notion_id = None
        self._selected_notion_type = None
        self._selected_recording_id = None
        self._notion_results = []
        
        # Load data on view
        self.after(100, self._initial_load)
    
    def _initial_load(self):
        """Load initial data when view is shown."""
        self._check_connection()
        self._load_recordings()
        # Auto-load databases after connection check
        self.after(800, self._auto_load_databases)
    
    def _auto_load_databases(self):
        """Auto-load databases on startup for intuitive first view."""
        self._filter_type("database")
    
    def _check_connection(self):
        """Check Notion API connection."""
        self.connection_label.configure(text="⏳ Checking connection...")
        
        def task():
            import os
            from dotenv import load_dotenv
            load_dotenv()
            
            if not os.getenv('NOTION_API_KEY') and not os.getenv('NOTION_TOKEN'):
                return {"connected": False, "error": "No API key configured"}
            
            try:
                from src.notion_sync import NotionSyncService
                sync = NotionSyncService()
                # Quick ping
                sync.client.users.me()
                return {"connected": True}
            except Exception as e:
                return {"connected": False, "error": str(e)[:50]}
        
        def done(result):
            if result.get("connected"):
                self.connection_label.configure(text="🟢 Connected to Notion", foreground="#27ae60")
            else:
                error = result.get("error", "Unknown error")
                self.connection_label.configure(text=f"🔴 {error}", foreground="#e74c3c")
        
        threading.Thread(target=lambda: self._run_task(task, done), daemon=True).start()
    
    def _run_task(self, task, callback):
        """Run task in thread and call callback on main thread."""
        try:
            result = task()
        except Exception as e:
            result = {"error": str(e)}
        self.after(0, lambda: callback(result))
    
    def _search_notion(self):
        """Search Notion workspace."""
        query = self.search_var.get().strip()
        if not query:
            messagebox.showinfo("Search", "Enter a search query")
            return
        
        self.connection_label.configure(text=f"🔍 Searching: {query}...")
        
        def task():
            try:
                from src.notion_sync import NotionSyncService
                sync = NotionSyncService()
                
                # Use Notion search API
                response = sync.client.search(
                    query=query,
                    page_size=20
                )
                
                results = []
                for item in response.get("results", []):
                    obj_type = item.get("object", "unknown")
                    
                    # Extract title
                    title = "Untitled"
                    if obj_type == "page":
                        props = item.get("properties", {})
                        for prop_name, prop_val in props.items():
                            if prop_val.get("type") == "title":
                                title_arr = prop_val.get("title", [])
                                if title_arr:
                                    title = title_arr[0].get("plain_text", "Untitled")
                                break
                    elif obj_type == "database":
                        title_arr = item.get("title", [])
                        if title_arr:
                            title = title_arr[0].get("plain_text", "Untitled")
                    
                    # Extract last edited
                    last_edited = item.get("last_edited_time", "")[:10]
                    
                    results.append({
                        "id": item["id"],
                        "title": title,
                        "type": obj_type,
                        "updated": last_edited,
                        "url": item.get("url", ""),
                    })
                
                return {"results": results}
            except Exception as e:
                return {"error": str(e)}
        
        def done(result):
            if result.get("error"):
                self.connection_label.configure(text=f"❌ {result['error'][:40]}")
                return
            
            self._notion_results = result.get("results", [])
            self._populate_notion_tree(self._notion_results)
            self.connection_label.configure(text=f"✅ Found {len(self._notion_results)} results")
        
        threading.Thread(target=lambda: self._run_task(task, done), daemon=True).start()
    
    def _filter_type(self, filter_type: str):
        """Filter results by type - always do fresh search for best UX."""
        self._search_by_type(filter_type)
    
    def _search_by_type(self, obj_type: str):
        """Search specifically for databases or pages."""
        self.connection_label.configure(text=f"🔍 Loading {obj_type}s...")
        
        def task():
            try:
                from src.notion_sync import NotionSyncService
                sync = NotionSyncService()
                
                # Search all objects, then filter client-side (Notion API doesn't support filter on object type)
                response = sync.client.search(page_size=50)
                
                # Filter to requested object type
                filtered_results = [item for item in response.get("results", []) if item.get("object") == obj_type]
                response["results"] = filtered_results
                
                results = []
                
                for item in response.get("results", []):
                    title = "Untitled"
                    if obj_type == "page":
                        props = item.get("properties", {})
                        for prop_name, prop_val in props.items():
                            if prop_val.get("type") == "title":
                                title_arr = prop_val.get("title", [])
                                if title_arr:
                                    title = title_arr[0].get("plain_text", "Untitled")
                                break
                    elif obj_type == "database":
                        title_arr = item.get("title", [])
                        if title_arr:
                            title = title_arr[0].get("plain_text", "Untitled")
                    
                    results.append({
                        "id": item["id"],
                        "title": title,
                        "type": obj_type,
                        "updated": item.get("last_edited_time", "")[:10],
                        "url": item.get("url", ""),
                    })
                
                return {"results": results}
            except Exception as e:
                return {"error": str(e)}
        
        def done(result):
            if result.get("error"):
                self.connection_label.configure(text=f"❌ {result['error'][:40]}")
                return
            
            results = result.get("results", [])
            # Filter out header rows
            actual_results = [r for r in results if r.get('type') != 'header']
            self._notion_results = actual_results
            self._populate_notion_tree(actual_results)
            
            # Clear status message
            count = len(actual_results)
            type_name = "Database" if obj_type == "database" else "Page"
            plural = "s" if count != 1 else ""
            icon = "📊" if obj_type == "database" else "📄"
            self.connection_label.configure(text=f"{icon} {count} {type_name}{plural} found", foreground="#27ae60")
        
        threading.Thread(target=lambda: self._run_task(task, done), daemon=True).start()
    
    def _show_recent(self):
        """Show recently edited pages."""
        self.connection_label.configure(text="🔍 Loading recent...")
        
        def task():
            try:
                from src.notion_sync import NotionSyncService
                sync = NotionSyncService()
                
                response = sync.client.search(
                    sort={"direction": "descending", "timestamp": "last_edited_time"},
                    page_size=30
                )
                
                results = []
                for item in response.get("results", []):
                    obj_type = item.get("object", "unknown")
                    title = "Untitled"
                    
                    if obj_type == "page":
                        props = item.get("properties", {})
                        for prop_val in props.values():
                            if prop_val.get("type") == "title":
                                title_arr = prop_val.get("title", [])
                                if title_arr:
                                    title = title_arr[0].get("plain_text", "Untitled")
                                break
                    elif obj_type == "database":
                        title_arr = item.get("title", [])
                        if title_arr:
                            title = title_arr[0].get("plain_text", "Untitled")
                    
                    results.append({
                        "id": item["id"],
                        "title": title,
                        "type": obj_type,
                        "updated": item.get("last_edited_time", "")[:10],
                        "url": item.get("url", ""),
                    })
                
                return {"results": results}
            except Exception as e:
                return {"error": str(e)}
        
        def done(result):
            if result.get("error"):
                self.connection_label.configure(text=f"❌ {result['error'][:40]}")
                return
            
            results = result.get("results", [])
            self._notion_results = results
            self._populate_notion_tree(results)
            self.connection_label.configure(text=f"✅ {len(results)} recent items")
        
        threading.Thread(target=lambda: self._run_task(task, done), daemon=True).start()
    
    def _show_linked(self):
        """Show recordings that are already linked to Notion."""
        # TODO: Query local DB for recordings with notion_page_id
        messagebox.showinfo("Linked", "Shows recordings already synced to Notion (coming soon)")
    
    def _populate_notion_tree(self, results: list):
        """Populate the Notion results tree."""
        for item in self.notion_tree.get_children():
            self.notion_tree.delete(item)
        
        for r in results:
            icon = "📚" if r["type"] == "database" else "📄"
            self.notion_tree.insert("", "end", iid=r["id"], values=(
                f"{icon} {r['title'][:40]}",
                r["type"],
                r["updated"]
            ))
    
    def _on_notion_select(self, event):
        """Handle Notion tree selection."""
        selection = self.notion_tree.selection()
        if selection:
            item_id = selection[0]
            # Find the item in results
            for r in self._notion_results:
                if r["id"] == item_id:
                    self._selected_notion_id = r["id"]
                    self._selected_notion_type = r["type"]
                    self.selected_notion.configure(text=f"{r['title'][:50]} ({r['type']})")
                    break
    
    def _on_notion_double_click(self, event):
        """Open Notion page in browser on double-click."""
        selection = self.notion_tree.selection()
        if selection:
            item_id = selection[0]
            for r in self._notion_results:
                if r["id"] == item_id and r.get("url"):
                    import webbrowser
                    webbrowser.open(r["url"])
                    break
    
    def _load_recordings(self):
        """Load recordings from local database."""
        def task():
            from src.database.engine import SessionLocal, init_db
            from src.database.models import Recording
            from sqlalchemy import select
            
            init_db()
            session = SessionLocal()
            try:
                recordings = session.execute(select(Recording)).scalars().all()
                return [{
                    "id": r.id,
                    "title": r.title or r.filename or "Untitled",
                    "created_at": str(r.created_at)[:10] if r.created_at else "",
                    "status": r.status,
                    "extra": r.extra or {},
                } for r in recordings]
            finally:
                session.close()
        
        def done(result):
            if isinstance(result, list):
                self._populate_recording_tree(result)
                # Also populate combo
                titles = [f"{r['title'][:40]} ({r['id'][:8]})" for r in result]
                self.recording_combo['values'] = titles
        
        threading.Thread(target=lambda: self._run_task(task, done), daemon=True).start()
    
    def _populate_recording_tree(self, recordings: list):
        """Populate the recording tree."""
        for item in self.recording_tree.get_children():
            self.recording_tree.delete(item)
        
        for r in recordings:
            # Check if linked to Notion
            notion_status = "✓" if r.get("extra", {}).get("notion_page_id") else "—"
            self.recording_tree.insert("", "end", iid=r["id"], values=(
                r["title"][:30],
                r["created_at"],
                notion_status
            ))
    
    def _on_recording_select(self, event):
        """Handle recording tree selection."""
        selection = self.recording_tree.selection()
        if selection:
            self._selected_recording_id = selection[0]
    
    def _link_recording(self):
        """Link selected recording to selected Notion page."""
        if not self._selected_notion_id:
            messagebox.showwarning("Link", "Select a Notion page first")
            return
        if not self._selected_recording_id:
            messagebox.showwarning("Link", "Select a recording first")
            return
        
        def task():
            from src.database.engine import SessionLocal, init_db
            from src.database.models import Recording
            import json
            
            init_db()
            session = SessionLocal()
            try:
                rec = session.get(Recording, self._selected_recording_id)
                if rec:
                    extra = rec.extra or {}
                    extra["notion_page_id"] = self._selected_notion_id
                    rec.extra = extra
                    session.commit()
                    return {"success": True, "title": rec.title}
                return {"error": "Recording not found"}
            finally:
                session.close()
        
        def done(result):
            if result.get("success"):
                messagebox.showinfo("Linked", f"Linked '{result['title']}' to Notion page")
                self._load_recordings()
            else:
                messagebox.showerror("Error", result.get("error", "Failed to link"))
        
        threading.Thread(target=lambda: self._run_task(task, done), daemon=True).start()
    
    def _push_to_page(self):
        """Push recording content to the selected Notion page."""
        if not self._selected_notion_id or self._selected_notion_type != "page":
            messagebox.showwarning("Push", "Select a Notion page first (not database)")
            return
        if not self._selected_recording_id:
            messagebox.showwarning("Push", "Select a recording first")
            return
        
        messagebox.showinfo("Push", "Push to specific page coming soon.\nUse 'Push All' to sync to default database.")
    
    def _create_notion_page(self):
        """Create a new Notion page for the selected recording."""
        if not self._selected_recording_id:
            messagebox.showwarning("Create", "Select a recording first")
            return
        
        # If a database is selected, create page in it
        if self._selected_notion_id and self._selected_notion_type == "database":
            parent_id = self._selected_notion_id
            parent_type = "database"
        else:
            parent_id = None
            parent_type = "workspace"
        
        def task():
            from src.database.engine import SessionLocal, init_db
            from src.database.models import Recording
            from src.notion_sync import NotionSyncService
            
            init_db()
            session = SessionLocal()
            try:
                rec = session.get(Recording, self._selected_recording_id)
                if not rec:
                    return {"error": "Recording not found"}
                
                sync = NotionSyncService()
                
                if parent_type == "database" and parent_id:
                    # Create in specific database
                    # This would need custom property mapping
                    success, result = sync.push_recording(rec)
                    if success:
                        return {"success": True, "page_id": result}
                    else:
                        return {"error": result}
                else:
                    # Create as standalone page
                    success, result = sync.push_recording(rec)
                    if success:
                        return {"success": True, "page_id": result}
                    else:
                        return {"error": result}
            finally:
                session.close()
        
        def done(result):
            if result.get("success"):
                messagebox.showinfo("Created", f"Created Notion page: {result.get('page_id', '')[:8]}...")
                self._load_recordings()
            else:
                messagebox.showerror("Error", result.get("error", "Failed to create"))
        
        threading.Thread(target=lambda: self._run_task(task, done), daemon=True).start()
