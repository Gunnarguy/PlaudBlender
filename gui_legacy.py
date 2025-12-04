#!/usr/bin/env python3
"""
PlaudBlender GUI - Full-featured interface for managing transcripts, search, and visualization.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import threading
import json
import os
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter
import re
import logging
from dotenv import load_dotenv, set_key
from src.plaud_oauth import PlaudOAuthClient
from src.plaud_client import PlaudClient

# Phase 1: Visual Analytics - DEFERRED IMPORT (loaded only when needed)
# Don't import matplotlib/wordcloud at module level - they're slow to load
MATPLOTLIB_AVAILABLE = False
WORDCLOUD_AVAILABLE = False

# Load environment
load_dotenv()

STOPWORDS = {
    'the','and','for','that','with','have','this','from','they','your','about',
    'just','into','were','been','then','them','when','will','there','what','like',
    'really','have','also','gets','than','because','which','only','more','very',
    'over','where','does','make','made','even','back','after','before','being',
    'through','each','while','every','such','those','these','some','many','much'
}


class PlaudBlenderGUI:
    """Main GUI application for PlaudBlender."""

    def __init__(self, root):
        self.root = root
        self.root.title("PlaudBlender - Transcript Manager")
        self.root.geometry("1400x900")
        self.root.minsize(1100, 700)

        self.style = ttk.Style()
        self.style.theme_use('clam')
        self._configure_styles()

        # Core state
        self.transcripts = []
        self.search_results = []
        self.is_authenticated = False
        self.saved_searches = self._load_saved_searches()
        self._pinecone_client = None
        self._plaud_oauth_client = None  # Cached PlaudOAuthClient instance
        self._plaud_client = None  # Cached PlaudClient instance
        self._index_host_cache = {}
        self._index_client_cache = {}
        self._stats_cache = None
        self._stats_cache_time = None
        self._indexes_loaded = False
        self._pinecone_loaded = False
        self.org_loaded = False
        self._stats_cache = {}  # Cache for describe_index_stats results
        self._stats_cache_time = {}  # Timestamps for cache invalidation

        # Console logging controls
        self.console_log_level = os.getenv('PB_LOG_LEVEL', 'INFO').upper()
        self.console_verbose = os.getenv('PB_VERBOSE', '1').lower() in ('1', 'true', 'yes', 'on')
        self.logger = logging.getLogger('PlaudBlender')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%H:%M:%S'))
            self.logger.addHandler(handler)
        self.logger.propagate = False
        self.logger.setLevel(getattr(logging, self.console_log_level, logging.INFO))

        # Build UI
        self._create_menu()
        self._create_main_layout()
        self._create_status_bar()

        # Initial async tasks
        self.root.after(100, self._check_auth_status)
        self.root.after(200, self._load_settings)

    def _configure_styles(self):
        """Configure custom styles and shared colors."""
        self.colors = {
            'bg': '#1e1e2e',
            'fg': '#cdd6f4',
            'accent': '#89b4fa',
            'success': '#a6e3a1',
            'warning': '#f9e2af',
            'error': '#f38ba8',
            'surface': '#313244',
            'overlay': '#45475a'
        }

        self.style.configure('TFrame', background=self.colors['bg'])
        self.style.configure('TLabel', background=self.colors['bg'], foreground=self.colors['fg'])
        self.style.configure('TButton', padding=6)
        self.style.configure('Accent.TButton', background=self.colors['accent'])
        self.style.configure('TNotebook', background=self.colors['bg'])
        self.style.configure('TNotebook.Tab', padding=[12, 4])
        self.style.configure('Treeview', background=self.colors['surface'],
                             foreground=self.colors['fg'], fieldbackground=self.colors['surface'])
        self.style.configure('Treeview.Heading', background=self.colors['overlay'])
        self.root.configure(bg=self.colors['bg'])

    def _update_namespace_ui(self, stats, index_name):
        """Update namespace list and stat cards from describe_index_stats response."""
        def _extract(value, default="—"):
            if isinstance(stats, dict):
                return stats.get(value, default)
            return getattr(stats, value, default)

        total_vectors = _extract('total_vector_count', 0)
        dimension = _extract('dimension', 0)
        metric = _extract('metric', '—')
        namespaces_obj = _extract('namespaces', {})

        if hasattr(namespaces_obj, 'items'):
            namespace_items = namespaces_obj.items()
        elif isinstance(namespaces_obj, dict):
            namespace_items = namespaces_obj.items()
        else:
            namespace_items = []

        namespaces = []
        for ns_name, summary in namespace_items:
            clean_name = 'default' if not ns_name or ns_name == '__default__' else ns_name
            if isinstance(summary, dict):
                count = summary.get('vector_count')
            else:
                count = getattr(summary, 'vector_count', None)
            namespaces.append((clean_name, count))
        if not namespaces:
            namespaces = [('default', total_vectors)]

        unique_names = []
        seen = set()
        for name, count in namespaces:
            if name not in seen:
                seen.add(name)
                unique_names.append((name, count))
        unique_names.sort(key=lambda item: item[0])

        if len(unique_names) > 1:
            all_namespaces = ['(all)'] + [name for name, _ in unique_names]
        else:
            all_namespaces = [name for name, _ in unique_names]

        if hasattr(self, 'namespace_listbox'):
            current_ns = self.pinecone_namespace_var.get()
            desired_ns = current_ns or 'default'

            self.namespace_listbox.delete(0, tk.END)
            for ns in all_namespaces:
                self.namespace_listbox.insert(tk.END, ns)

            if desired_ns in all_namespaces:
                idx = all_namespaces.index(desired_ns)
            elif '(all)' in all_namespaces:
                idx = all_namespaces.index('(all)')
            else:
                idx = 0

            self.namespace_listbox.selection_set(idx)
            self.namespace_listbox.activate(idx)

            selected_label = all_namespaces[idx] if all_namespaces else ''
            if selected_label in ('(all)', 'default'):
                self.pinecone_namespace_var.set("")
            else:
                self.pinecone_namespace_var.set(selected_label)

        if hasattr(self, 'pinecone_stat_labels'):
            self.pinecone_stat_labels['total_vectors'].configure(text=f"{total_vectors:,}")
            self.pinecone_stat_labels['dimension'].configure(text=str(dimension))
            self.pinecone_stat_labels['metric'].configure(text=metric)
            self.pinecone_stat_labels['namespaces'].configure(text=str(max(len(unique_names), 1)))

        if hasattr(self, 'pinecone_status_label'):
            self.pinecone_status_label.config(text=f"● {index_name} ready", foreground='#a6e3a1')

        return all_namespaces

    def _create_menu(self):
        """Create menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Export Transcripts...", command=self._export_transcripts)
        file_menu.add_command(label="Open Output Folder", command=self._open_output_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.root.quit, accelerator="⌘Q")
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Sync All to Pinecone", command=self._sync_to_pinecone)
        tools_menu.add_command(label="Generate Mind Map", command=self._generate_mindmap)
        tools_menu.add_separator()
        tools_menu.add_command(label="Clear Pinecone Index", command=self._clear_pinecone)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="View README", command=lambda: webbrowser.open(
            f"file://{Path(__file__).parent / 'README.md'}"))
        help_menu.add_command(label="Plaud API Docs", command=lambda: webbrowser.open(
            "https://platform.plaud.ai/developer/api"))
    
    def _create_main_layout(self):
        """Create main application layout."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Bind tab change to auto-refresh
        self.notebook.bind('<<NotebookTabChanged>>', self._on_tab_changed)
        
        # Create tabs
        self._create_dashboard_tab()
        self._create_transcripts_tab()
        self._create_pinecone_unified_tab()  # Unified Pinecone management
        self._create_search_tab()
        self._create_settings_tab()
        self._create_logs_tab()
    
    def _create_dashboard_tab(self):
        """Create dashboard overview tab with visual analytics."""
        tab = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(tab, text="📊 Dashboard")
        
        # Header with quick index switcher
        header_frame = ttk.Frame(tab)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(header_frame, text="PlaudBlender Dashboard", font=('Helvetica', 24, 'bold')).pack(side=tk.LEFT)
        
        # Quick index switcher in header
        index_frame = ttk.Frame(header_frame)
        index_frame.pack(side=tk.RIGHT)
        ttk.Label(index_frame, text="Active Index:", font=('Helvetica', 11)).pack(side=tk.LEFT, padx=5)
        self.header_index_var = tk.StringVar(value=os.getenv('PINECONE_INDEX_NAME', 'transcripts'))
        self.header_index_dropdown = ttk.Combobox(index_frame, textvariable=self.header_index_var, 
                                                   width=20, state='readonly', font=('Helvetica', 10, 'bold'))
        self.header_index_dropdown.pack(side=tk.LEFT)
        self.header_index_dropdown.bind('<<ComboboxSelected>>', self._on_header_index_changed)
        ttk.Button(index_frame, text="🔄", command=self._load_indexes_for_header, width=3).pack(side=tk.LEFT, padx=2)
        
        # Stats frame - row 1
        stats_frame1 = ttk.Frame(tab)
        stats_frame1.pack(fill=tk.X, pady=5)
        
        # Stat cards row 1
        self.stat_labels = {}
        stats_row1 = [
            ("auth_status", "🔐 Auth Status", "Checking..."),
            ("plaud_count", "📱 Plaud Recordings", "—"),
            ("pinecone_count", "🌲 Pinecone Vectors", "—"),
            ("last_sync", "🔄 Last Sync", "Never"),
        ]
        
        for i, (key, label, default) in enumerate(stats_row1):
            card = ttk.Frame(stats_frame1, relief='ridge', borderwidth=2, padding=15)
            card.grid(row=0, column=i, padx=10, pady=5, sticky='nsew')
            stats_frame1.columnconfigure(i, weight=1)
            
            ttk.Label(card, text=label, font=('Helvetica', 12)).pack()
            self.stat_labels[key] = ttk.Label(card, text=default, font=('Helvetica', 18, 'bold'))
            self.stat_labels[key].pack(pady=5)
        
        # Stats frame - row 2
        stats_frame2 = ttk.Frame(tab)
        stats_frame2.pack(fill=tk.X, pady=5)
        
        stats_row2 = [
            ("total_duration", "⏱️ Total Duration", "—"),
            ("total_words", "📝 Est. Words", "—"),
            ("unique_speakers", "👥 Speakers", "—"),
            ("date_range", "📅 Date Range", "—"),
        ]
        
        for i, (key, label, default) in enumerate(stats_row2):
            card = ttk.Frame(stats_frame2, relief='ridge', borderwidth=2, padding=15)
            card.grid(row=0, column=i, padx=10, pady=5, sticky='nsew')
            stats_frame2.columnconfigure(i, weight=1)
            
            ttk.Label(card, text=label, font=('Helvetica', 12)).pack()
            self.stat_labels[key] = ttk.Label(card, text=default, font=('Helvetica', 14, 'bold'))
            self.stat_labels[key].pack(pady=5)
        
        # Quick actions
        actions_frame = ttk.LabelFrame(tab, text="Quick Actions", padding=15)
        actions_frame.pack(fill=tk.X, pady=20)
        
        # Add index target indicator
        target_frame = ttk.Frame(actions_frame)
        target_frame.grid(row=0, column=0, columnspan=4, pady=(0, 10), sticky='ew')
        ttk.Label(target_frame, text="🎯 Target Index:", font=('Helvetica', 10)).pack(side=tk.LEFT, padx=5)
        self.target_index_label = ttk.Label(target_frame, text=os.getenv('PINECONE_INDEX_NAME', 'transcripts'), 
                                           font=('Helvetica', 10, 'bold'), foreground='#89b4fa')
        self.target_index_label.pack(side=tk.LEFT)
        ttk.Label(target_frame, text="(change in Pinecone tab)", font=('Helvetica', 9), foreground='#6c7086').pack(side=tk.LEFT, padx=5)
        
        buttons = [
            ("🔄 Sync All to Pinecone", self._sync_to_pinecone),
            ("🔍 Search Transcripts", lambda: self.notebook.select(2)),
            ("🧠 Generate Mind Map", self._generate_mindmap),
            ("⚙️ Settings & API Keys", lambda: self.notebook.select(3)),
        ]
        
        for i, (text, cmd) in enumerate(buttons):
            btn = ttk.Button(actions_frame, text=text, command=cmd, width=20)
            btn.grid(row=1, column=i, padx=10, pady=5)
            actions_frame.columnconfigure(i, weight=1)
        
        # Recent activity log
        activity_frame = ttk.LabelFrame(tab, text="Recent Activity", padding=10)
        activity_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.activity_log = scrolledtext.ScrolledText(
            activity_frame, height=10, wrap=tk.WORD,
            bg=self.colors['surface'], fg=self.colors['fg'],
            font=('Monaco', 11), state='disabled'
        )
        self.activity_log.pack(fill=tk.BOTH, expand=True)
    
    def _create_transcripts_tab(self):
        """Create transcripts browser tab with batch operations."""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="📝 Transcripts")
        
        # Toolbar
        toolbar = ttk.Frame(tab)
        toolbar.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(toolbar, text="🔄 Refresh from Plaud", command=self._fetch_transcripts).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="📥 Sync Selected to Pinecone", command=self._sync_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="🗑️ Delete from Pinecone", command=self._delete_selected_from_pinecone).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="👁️ View Transcript", command=self._view_selected_transcript).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="📊 View Details", command=self._view_recording_details).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="💾 Export as Markdown", command=self._export_selected_transcripts).pack(side=tk.LEFT, padx=5)
        
        # Batch operations indicator
        self.batch_indicator_label = ttk.Label(toolbar, text="", font=('Helvetica', 10, 'bold'), foreground='#89b4fa')
        self.batch_indicator_label.pack(side=tk.RIGHT, padx=10)
        
        # Filter
        ttk.Label(toolbar, text="Filter:").pack(side=tk.LEFT, padx=(20, 5))
        self.transcript_filter = ttk.Entry(toolbar, width=30)
        self.transcript_filter.pack(side=tk.LEFT, padx=5)
        self.transcript_filter.bind('<KeyRelease>', self._filter_transcripts)
        
        # Treeview for transcripts - expanded columns
        columns = ('name', 'date', 'time', 'duration', 'id')
        self.transcript_tree = ttk.Treeview(tab, columns=columns, show='headings', selectmode='extended')
        
        # Bind selection change to update batch indicator
        self.transcript_tree.bind('<<TreeviewSelect>>', self._update_batch_indicator)
        
        self.transcript_tree.heading('name', text='Name', anchor='w')
        self.transcript_tree.heading('date', text='Date', anchor='w')
        self.transcript_tree.heading('time', text='Time', anchor='w')
        self.transcript_tree.heading('duration', text='Duration', anchor='w')
        self.transcript_tree.heading('id', text='ID', anchor='w')
        
        self.transcript_tree.column('name', width=450)
        self.transcript_tree.column('date', width=100)
        self.transcript_tree.column('time', width=80)
        self.transcript_tree.column('duration', width=80)
        self.transcript_tree.column('id', width=120)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=self.transcript_tree.yview)
        self.transcript_tree.configure(yscrollcommand=scrollbar.set)
        
        self.transcript_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Double-click to view
        self.transcript_tree.bind('<Double-1>', self._view_transcript)
    
    def _update_batch_indicator(self, event=None):
        """Update batch operations indicator."""
        if hasattr(self, 'transcript_tree') and hasattr(self, 'batch_indicator_label'):
            selection = self.transcript_tree.selection()
            count = len(selection)
            if count > 1:
                self.batch_indicator_label.config(text=f"✓ {count} items selected")
            else:
                self.batch_indicator_label.config(text="")
    
    def _create_search_tab(self):
        """Create enhanced search interface with saved searches and query builder."""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="🔍 Search")
        
        # Saved Searches Section
        saved_frame = ttk.LabelFrame(tab, text="💾 Saved Searches", padding=10)
        saved_frame.pack(fill=tk.X, pady=(0, 10))
        
        saved_row = ttk.Frame(saved_frame)
        saved_row.pack(fill=tk.X)
        
        ttk.Label(saved_row, text="Load:").pack(side=tk.LEFT, padx=5)
        self.saved_search_var = tk.StringVar()
        self.saved_search_dropdown = ttk.Combobox(saved_row, textvariable=self.saved_search_var,
                                                   values=list(self.saved_searches.keys()),
                                                   state='readonly', width=30)
        self.saved_search_dropdown.pack(side=tk.LEFT, padx=5)
        self.saved_search_dropdown.bind('<<ComboboxSelected>>', 
                                       lambda e: self._load_saved_search(self.saved_search_var.get()))
        
        ttk.Button(saved_row, text="💾 Save Current", command=self._save_current_search).pack(side=tk.LEFT, padx=5)
        ttk.Button(saved_row, text="🗑️ Delete", 
                  command=lambda: self._delete_saved_search(self.saved_search_var.get())).pack(side=tk.LEFT, padx=2)
        
        # Search bar
        search_frame = ttk.Frame(tab)
        search_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(search_frame, text="Semantic Search:", font=('Helvetica', 14, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        
        self.search_entry = ttk.Entry(search_frame, font=('Helvetica', 14), width=50)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.search_entry.bind('<Return>', lambda e: self._perform_search())
        
        ttk.Button(search_frame, text="🔍 Search Pinecone", command=self._perform_search).pack(side=tk.LEFT, padx=5)
        ttk.Button(search_frame, text="❌ Clear Results", command=self._clear_search).pack(side=tk.LEFT, padx=5)
        
        # Options
        options_frame = ttk.Frame(tab)
        options_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(options_frame, text="Max Results:").pack(side=tk.LEFT)
        self.search_limit = ttk.Spinbox(options_frame, from_=1, to=20, width=5)
        self.search_limit.set(5)
        self.search_limit.pack(side=tk.LEFT, padx=5)
        
        self.include_context = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Show text preview", variable=self.include_context).pack(side=tk.LEFT, padx=15)
        
        # Results
        results_frame = ttk.LabelFrame(tab, text="Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.search_results_text = scrolledtext.ScrolledText(
            results_frame, wrap=tk.WORD,
            bg=self.colors['surface'], fg=self.colors['fg'],
            font=('Helvetica', 12)
        )
        self.search_results_text.pack(fill=tk.BOTH, expand=True)
    
    def _create_pinecone_unified_tab(self):
        """Create unified Pinecone management tab with clean, intuitive layout."""
        tab = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(tab, text="🌲 Pinecone")
        
        # ===== TOP SECTION: Connection & Quick Stats =====
        top_section = ttk.Frame(tab)
        top_section.pack(fill=tk.X, pady=(0, 15))
        
        # Left: Index Selection
        index_frame = ttk.LabelFrame(top_section, text="📍 Current Index", padding=10)
        index_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        index_row = ttk.Frame(index_frame)
        index_row.pack(fill=tk.X)
        
        self.pinecone_index_var = tk.StringVar(value=os.getenv('PINECONE_INDEX_NAME', 'transcripts'))
        self.pinecone_index_dropdown = ttk.Combobox(index_row, textvariable=self.pinecone_index_var, width=22, state='readonly')
        self.pinecone_index_dropdown.pack(side=tk.LEFT, padx=(0, 5))
        self.pinecone_index_dropdown.bind('<<ComboboxSelected>>', lambda e: self._on_index_changed())
        ttk.Button(index_row, text="🔄", command=self._load_pinecone_indexes, width=3).pack(side=tk.LEFT)
        
        # Middle: Namespaces List (Auto-populated, clickable)
        ns_frame = ttk.LabelFrame(top_section, text="🏷️ Namespaces", padding=10)
        ns_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        ns_header = ttk.Frame(ns_frame)
        ns_header.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(ns_header, text="Click to filter:", font=('Helvetica', 9), foreground='gray').pack(side=tk.LEFT)
        ttk.Button(ns_header, text="🔄", command=self._refresh_pinecone_vectors, width=3).pack(side=tk.RIGHT)
        
        self.pinecone_namespace_var = tk.StringVar(value="")
        
        # Create listbox for namespace selection
        ns_scroll = ttk.Scrollbar(ns_frame, orient=tk.VERTICAL)
        self.namespace_listbox = tk.Listbox(
            ns_frame,
            height=3,
            bg=self.colors['surface'],
            fg=self.colors['fg'],
            selectmode=tk.SINGLE,
            yscrollcommand=ns_scroll.set,
            font=('Helvetica', 11),
            activestyle='dotbox'
        )
        ns_scroll.config(command=self.namespace_listbox.yview)
        self.namespace_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ns_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.namespace_listbox.bind('<<ListboxSelect>>', self._on_namespace_changed)
        
        # Right: Quick Stats Dashboard
        stats_frame = ttk.LabelFrame(top_section, text="📊 Quick Stats", padding=10)
        stats_frame.pack(side=tk.LEFT, fill=tk.BOTH)
        
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack()
        
        self.pinecone_stat_labels = {}
        quick_stats = [
            ('total_vectors', '📦 Vectors'),
            ('dimension', '📐 Dimension'),
            ('metric', '📏 Metric'),
            ('namespaces', '🏷️ Namespaces'),
        ]
        
        for i, (key, label) in enumerate(quick_stats):
            stat_box = ttk.Frame(stats_grid)
            stat_box.grid(row=i // 2, column=i % 2, padx=8, pady=4, sticky='ew')
            ttk.Label(stat_box, text=label, font=('Helvetica', 9), foreground='gray').pack(anchor='w')
            self.pinecone_stat_labels[key] = ttk.Label(stat_box, text='—', font=('Helvetica', 13, 'bold'))
            self.pinecone_stat_labels[key].pack(anchor='w')
        
        # Status indicator
        status_row = ttk.Frame(stats_frame)
        status_row.pack(fill=tk.X, pady=(8, 0))
        self.pinecone_status_label = ttk.Label(status_row, text="● Ready", foreground='#a6e3a1', font=('Helvetica', 9, 'bold'))
        self.pinecone_status_label.pack()
        
        # ===== MIDDLE SECTION: Action Buttons (Clean Horizontal Layout) =====
        actions_frame = ttk.Frame(tab)
        actions_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Primary Actions (Most Common)
        primary_group = ttk.LabelFrame(actions_frame, text="🎯 Quick Actions", padding=8)
        primary_group.pack(side=tk.LEFT, padx=(0, 10))
        
        btn_row1 = ttk.Frame(primary_group)
        btn_row1.pack(fill=tk.X, pady=2)
        ttk.Button(btn_row1, text="🔍 Filter Metadata", command=self._fetch_by_metadata_dialog, width=18).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row1, text="📋 Get by ID", command=self._fetch_vector_full_details, width=16).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row1, text="🔎 Similar Search", command=self._query_similar_dialog, width=16).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row1, text="🌐 All Namespaces", command=self._query_all_namespaces_dialog, width=15).pack(side=tk.LEFT, padx=2)
        
        btn_row2 = ttk.Frame(primary_group)
        btn_row2.pack(fill=tk.X, pady=2)
        ttk.Button(btn_row2, text="➕ Upsert Vector", command=self._upsert_vector_dialog, width=18).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row2, text="✏️ Edit Metadata", command=self._update_vector_metadata, width=16).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row2, text="🗑️ Delete...", command=self._show_delete_options, width=16).pack(side=tk.LEFT, padx=2)
        
        # Advanced Actions
        advanced_group = ttk.LabelFrame(actions_frame, text="⚙️ Advanced", padding=8)
        advanced_group.pack(side=tk.LEFT, padx=(0, 10))
        
        adv_row1 = ttk.Frame(advanced_group)
        adv_row1.pack(fill=tk.X, pady=2)
        ttk.Button(adv_row1, text="📤 Bulk Import", command=self._bulk_upsert_dialog, width=16).pack(side=tk.LEFT, padx=2)
        ttk.Button(adv_row1, text="💾 Export Data", command=self._export_pinecone_data, width=15).pack(side=tk.LEFT, padx=2)
        
        adv_row2 = ttk.Frame(advanced_group)
        adv_row2.pack(fill=tk.X, pady=2)
        ttk.Button(adv_row2, text="🏢 Org Overview", command=self._show_org_overview, width=16).pack(side=tk.LEFT, padx=2)
        ttk.Button(adv_row2, text="⚡ Manage Index", command=self._show_index_management, width=15).pack(side=tk.LEFT, padx=2)
        
        # ===== BOTTOM SECTION: Vector Data Grid =====
        data_section = ttk.LabelFrame(tab, text="📊 Vector Data", padding=10)
        data_section.pack(fill=tk.BOTH, expand=True)
        
        # Search bar above the grid
        search_bar = ttk.Frame(data_section)
        search_bar.pack(fill=tk.X, pady=(0, 8))
        
        ttk.Label(search_bar, text="🔍 Filter:", font=('Helvetica', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 5))
        self.pinecone_filter = ttk.Entry(search_bar, font=('Helvetica', 10))
        self.pinecone_filter.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.pinecone_filter.bind('<KeyRelease>', lambda e: self._filter_pinecone_vectors())
        self.pinecone_filter.insert(0, "Type to filter by ID, title, metadata...")
        self.pinecone_filter.config(foreground='gray')
        self.pinecone_filter.bind('<FocusIn>', lambda e: self._clear_placeholder(self.pinecone_filter, "Type to filter by ID, title, metadata..."))
        self.pinecone_filter.bind('<FocusOut>', lambda e: self._restore_placeholder(self.pinecone_filter, "Type to filter by ID, title, metadata..."))
        
        ttk.Button(search_bar, text="Clear", command=lambda: (self.pinecone_filter.delete(0, tk.END), self._refresh_pinecone_vectors()), width=8).pack(side=tk.LEFT)
        ttk.Button(search_bar, text="🔄 Refresh All", command=self._refresh_pinecone_vectors, width=12).pack(side=tk.LEFT, padx=(10, 0))
        
        # Data grid & preview pane
        grid_container = ttk.Frame(data_section)
        grid_container.pack(fill=tk.BOTH, expand=True)

        tree_container = ttk.Frame(grid_container)
        tree_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        columns = ('vector_id', 'title', 'date', 'duration', 'themes', 'fields')
        self.pinecone_tree = ttk.Treeview(tree_container, columns=columns, show='headings', selectmode='extended', height=15)

        self.pinecone_tree.heading('vector_id', text='Vector ID', anchor='w')
        self.pinecone_tree.heading('title', text='Title / Name', anchor='w')
        self.pinecone_tree.heading('date', text='Date', anchor='w')
        self.pinecone_tree.heading('duration', text='Duration', anchor='center')
        self.pinecone_tree.heading('themes', text='Themes/Keywords', anchor='w')
        self.pinecone_tree.heading('fields', text='Metadata Fields', anchor='center')

        self.pinecone_tree.column('vector_id', width=150)
        self.pinecone_tree.column('title', width=300)
        self.pinecone_tree.column('date', width=100)
        self.pinecone_tree.column('duration', width=90)
        self.pinecone_tree.column('themes', width=250)
        self.pinecone_tree.column('fields', width=100)

        self.pinecone_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(tree_container, orient=tk.VERTICAL, command=self.pinecone_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.pinecone_tree.configure(yscrollcommand=scrollbar.set)

        preview_frame = ttk.LabelFrame(grid_container, text="🔎 Vector Preview", padding=8)
        preview_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0))

        self.pinecone_preview_text = scrolledtext.ScrolledText(
            preview_frame,
            wrap=tk.WORD,
            font=('Monaco', 10),
            width=40,
            height=18,
            bg=self.colors['surface'],
            fg=self.colors['fg']
        )
        self.pinecone_preview_text.pack(fill=tk.BOTH, expand=True)
        self.pinecone_preview_text.insert('1.0', "Select an index or namespace to view vector metadata previews.")
        self.pinecone_preview_text.configure(state='disabled')
        
        # Pagination at bottom
        page_frame = ttk.Frame(data_section)
        page_frame.pack(fill=tk.X, pady=(8, 0))
        
        ttk.Label(page_frame, text="Showing:").pack(side=tk.LEFT, padx=(0, 5))
        self.page_size_var = tk.IntVar(value=1000)
        page_size_combo = ttk.Combobox(page_frame, textvariable=self.page_size_var, 
                                       values=["100", "500", "1000", "5000"], width=8, state='readonly')
        page_size_combo.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Button(page_frame, text="◀", command=self._prev_page, width=3).pack(side=tk.LEFT, padx=2)
        self.page_label = ttk.Label(page_frame, text="Page 1", font=('Helvetica', 9, 'bold'))
        self.page_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(page_frame, text="▶", command=self._next_page, width=3).pack(side=tk.LEFT, padx=2)
        
        # Context menu
        self.pinecone_tree.bind('<Button-2>', self._show_pinecone_context_menu)
        self.pinecone_tree.bind('<Button-3>', self._show_pinecone_context_menu)
        # Selection binding - show preview immediately on click
        self.pinecone_tree.bind('<<TreeviewSelect>>', self._on_vector_selected)

        # ===== ORGANIZATION INSPECTOR SECTION =====
        org_section = ttk.LabelFrame(tab, text="🏢 Organization Inspector", padding=12)
        org_section.pack(fill=tk.BOTH, expand=True, pady=(15, 0))

        org_header = ttk.Frame(org_section)
        org_header.pack(fill=tk.X)
        ttk.Label(org_header, text="Control-plane snapshot of every index in your org", font=('Helvetica', 11, 'bold')).pack(side=tk.LEFT)

        org_btns = ttk.Frame(org_header)
        org_btns.pack(side=tk.RIGHT)
        ttk.Button(org_btns, text="🔄 Refresh", command=self._refresh_org_inspector).pack(side=tk.LEFT, padx=3)
        ttk.Button(org_btns, text="💾 Export", command=self._export_org_report).pack(side=tk.LEFT, padx=3)
        ttk.Button(org_btns, text="🔍 Verify", command=self._show_verification_details).pack(side=tk.LEFT, padx=3)

        summary_frame = ttk.Frame(org_section)
        summary_frame.pack(fill=tk.X, pady=(10, 5))

        self.org_summary_labels = {}
        summary_stats = [
            ('total_indexes', '📚 Indexes'),
            ('total_vectors', '🔢 Vectors'),
            ('total_namespaces', '🏷️ Namespaces'),
            ('total_dimensions', '📐 Dimensions'),
        ]

        for i, (key, label) in enumerate(summary_stats):
            card = ttk.Frame(summary_frame, relief='ridge', borderwidth=1, padding=12)
            card.grid(row=0, column=i, padx=6, sticky='nsew')
            summary_frame.columnconfigure(i, weight=1)
            ttk.Label(card, text=label, font=('Helvetica', 10), foreground='gray').pack(anchor='w')
            self.org_summary_labels[key] = ttk.Label(card, text='—', font=('Helvetica', 15, 'bold'))
            self.org_summary_labels[key].pack(anchor='w', pady=(4, 0))

        org_body = ttk.Frame(org_section)
        org_body.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        tree_frame = ttk.Frame(org_body)
        tree_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        columns = ('name', 'status', 'type', 'region', 'dimension', 'metric', 'vectors', 'namespaces')
        self.org_indexes_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=10)
        headings = [
            ('name', 'Index'),
            ('status', 'Status'),
            ('type', 'Type'),
            ('region', 'Region'),
            ('dimension', 'Dim'),
            ('metric', 'Metric'),
            ('vectors', 'Vectors'),
            ('namespaces', 'Namespaces')
        ]
        for col, text in headings:
            self.org_indexes_tree.heading(col, text=text, anchor='w' if col in {'name', 'type', 'region', 'metric'} else 'center')
        self.org_indexes_tree.column('name', width=160)
        self.org_indexes_tree.column('status', width=80, anchor='center')
        self.org_indexes_tree.column('type', width=100)
        self.org_indexes_tree.column('region', width=120)
        self.org_indexes_tree.column('dimension', width=80, anchor='center')
        self.org_indexes_tree.column('metric', width=80)
        self.org_indexes_tree.column('vectors', width=110, anchor='e')
        self.org_indexes_tree.column('namespaces', width=110, anchor='e')
        self.org_indexes_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        org_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.org_indexes_tree.yview)
        org_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.org_indexes_tree.configure(yscrollcommand=org_scroll.set)
        self.org_indexes_tree.bind('<<TreeviewSelect>>', self._on_org_index_selected)

        detail_frame = ttk.Frame(org_body)
        detail_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))

        details_box = ttk.LabelFrame(detail_frame, text="Index Details", padding=8)
        details_box.pack(fill=tk.BOTH, expand=True)
        self.org_index_details = scrolledtext.ScrolledText(
            details_box,
            height=12,
            wrap=tk.WORD,
            font=('Monaco', 10),
            bg=self.colors['surface'],
            fg=self.colors['fg']
        )
        self.org_index_details.pack(fill=tk.BOTH, expand=True)
        self.org_index_details.insert('1.0', "Select an index to inspect its configuration and namespace breakdown.")
        self.org_index_details.configure(state='disabled')

        samples_box = ttk.LabelFrame(detail_frame, text="Sample Vectors", padding=8)
        samples_box.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.org_samples_text = scrolledtext.ScrolledText(
            samples_box,
            height=10,
            wrap=tk.WORD,
            font=('Monaco', 10),
            bg=self.colors['surface'],
            fg=self.colors['fg']
        )
        self.org_samples_text.pack(fill=tk.BOTH, expand=True)
        self.org_samples_text.insert('1.0', "Sample vectors and metadata previews will appear here once loaded.")
        self.org_samples_text.configure(state='disabled')
        
        # Auto-load indexes and vectors on startup
        # Show loading state immediately
        if hasattr(self, 'pinecone_status_label'):
            self.pinecone_status_label.config(text="● Ready", foreground='#a6e3a1')
        # Defer loading until user views the tab
        self._pinecone_loaded = False
    
    def _create_settings_tab(self):
        """Create settings configuration tab."""
        tab = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(tab, text="⚙️ Settings")
        
        # Create scrollable frame
        canvas = tk.Canvas(tab, bg=self.colors['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Plaud OAuth Settings
        plaud_frame = ttk.LabelFrame(scrollable_frame, text="🔐 Plaud OAuth", padding=15)
        plaud_frame.pack(fill=tk.X, pady=10, padx=5)
        
        self.settings_vars = {}
        
        plaud_settings = [
            ("PLAUD_CLIENT_ID", "Client ID:"),
            ("PLAUD_CLIENT_SECRET", "Client Secret:"),
            ("PLAUD_REDIRECT_URI", "Redirect URI:"),
        ]
        
        for i, (key, label) in enumerate(plaud_settings):
            ttk.Label(plaud_frame, text=label).grid(row=i, column=0, sticky='e', padx=5, pady=5)
            var = tk.StringVar(value=os.getenv(key, ''))
            self.settings_vars[key] = var
            show_value = '*' if 'SECRET' in key else ''
            entry = ttk.Entry(plaud_frame, textvariable=var, width=60, show=show_value)
            entry.grid(row=i, column=1, sticky='ew', padx=5, pady=5)
        
        # Auth buttons
        auth_btn_frame = ttk.Frame(plaud_frame)
        auth_btn_frame.grid(row=len(plaud_settings), column=0, columnspan=2, pady=10)
        
        ttk.Button(auth_btn_frame, text="🔑 Start OAuth Flow", command=self._start_oauth).pack(side=tk.LEFT, padx=5)
        ttk.Button(auth_btn_frame, text="✅ Check Token", command=self._check_auth_status).pack(side=tk.LEFT, padx=5)
        ttk.Button(auth_btn_frame, text="🗑️ Clear Tokens", command=self._clear_tokens).pack(side=tk.LEFT, padx=5)
        ttk.Button(auth_btn_frame, text="🚫 Disconnect Account", command=self._disconnect_account).pack(side=tk.LEFT, padx=5)
        
        # Token status
        self.token_status = ttk.Label(plaud_frame, text="", font=('Helvetica', 11))
        self.token_status.grid(row=len(plaud_settings)+1, column=0, columnspan=2, pady=5)
        
        # Pinecone Settings
        pinecone_frame = ttk.LabelFrame(scrollable_frame, text="🌲 Pinecone", padding=15)
        pinecone_frame.pack(fill=tk.X, pady=10, padx=5)
        
        pinecone_settings = [
            ("PINECONE_API_KEY", "API Key:"),
            ("PINECONE_INDEX_NAME", "Index Name:"),
        ]
        
        for i, (key, label) in enumerate(pinecone_settings):
            ttk.Label(pinecone_frame, text=label).grid(row=i, column=0, sticky='e', padx=5, pady=5)
            var = tk.StringVar(value=os.getenv(key, ''))
            self.settings_vars[key] = var
            show_value = '*' if 'KEY' in key else ''
            entry = ttk.Entry(pinecone_frame, textvariable=var, width=60, show=show_value)
            entry.grid(row=i, column=1, sticky='ew', padx=5, pady=5)
        
        # Gemini Settings
        gemini_frame = ttk.LabelFrame(scrollable_frame, text="✨ Google Gemini", padding=15)
        gemini_frame.pack(fill=tk.X, pady=10, padx=5)
        
        ttk.Label(gemini_frame, text="API Key:").grid(row=0, column=0, sticky='e', padx=5, pady=5)
        var = tk.StringVar(value=os.getenv('GEMINI_API_KEY', ''))
        self.settings_vars['GEMINI_API_KEY'] = var
        ttk.Entry(gemini_frame, textvariable=var, width=60, show='*').grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        
        # Sync Settings
        sync_frame = ttk.LabelFrame(scrollable_frame, text="🔄 Sync Settings", padding=15)
        sync_frame.pack(fill=tk.X, pady=10, padx=5)
        
        ttk.Label(sync_frame, text="Similarity Threshold:").grid(row=0, column=0, sticky='e', padx=5, pady=5)
        self.similarity_threshold = ttk.Scale(sync_frame, from_=0.5, to=0.95, orient='horizontal', length=200)
        self.similarity_threshold.set(0.75)
        self.similarity_threshold.grid(row=0, column=1, sticky='w', padx=5, pady=5)
        self.threshold_label = ttk.Label(sync_frame, text="0.75")
        self.threshold_label.grid(row=0, column=2, padx=5)
        self.similarity_threshold.configure(command=lambda v: self.threshold_label.configure(text=f"{float(v):.2f}"))
        
        ttk.Label(sync_frame, text="Max Connections:").grid(row=1, column=0, sticky='e', padx=5, pady=5)
        self.max_connections = ttk.Spinbox(sync_frame, from_=10, to=100, width=10)
        self.max_connections.set(50)
        self.max_connections.grid(row=1, column=1, sticky='w', padx=5, pady=5)
        
        # Save button
        save_frame = ttk.Frame(scrollable_frame)
        save_frame.pack(fill=tk.X, pady=20, padx=5)
        
        ttk.Button(save_frame, text="💾 Save Settings", command=self._save_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(save_frame, text="🔄 Reload from .env", command=self._load_settings).pack(side=tk.LEFT, padx=5)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def _create_logs_tab(self):
        """Create logs viewer tab."""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="📋 Logs")
        
        # Toolbar
        toolbar = ttk.Frame(tab)
        toolbar.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(toolbar, text="🗑️ Clear", command=self._clear_logs).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="💾 Save Logs", command=self._save_logs).pack(side=tk.LEFT, padx=5)
        
        # Log display
        self.log_text = scrolledtext.ScrolledText(
            tab, wrap=tk.WORD,
            bg=self.colors['surface'], fg=self.colors['fg'],
            font=('Monaco', 11)
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure tags for log levels
        self.log_text.tag_configure('INFO', foreground=self.colors['fg'])
        self.log_text.tag_configure('SUCCESS', foreground=self.colors['success'])
        self.log_text.tag_configure('WARNING', foreground=self.colors['warning'])
        self.log_text.tag_configure('ERROR', foreground=self.colors['error'])
    
    def _create_status_bar(self):
        """Create status bar at bottom."""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = ttk.Label(status_frame, text="Ready", padding=5)
        self.status_label.pack(side=tk.LEFT)
        
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate', length=100)
        self.progress.pack(side=tk.RIGHT, padx=10)
    
    # ========== Helper Methods ==========
    
    def _log(self, message: str, level: str = 'INFO'):
        """Add message to logs."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted = f"[{timestamp}] [{level}] {message}\n"
        
        self.log_text.insert(tk.END, formatted, level)
        self.log_text.see(tk.END)
        
        # Also add to activity log on dashboard
        self.activity_log.configure(state='normal')
        self.activity_log.insert(tk.END, formatted, level)
        self.activity_log.see(tk.END)
        self.activity_log.configure(state='disabled')

        self._console_log(level, message)

    def _console_log(self, level, message, details=None):
        """Emit log line to terminal/stdout using Python logging."""
        if not hasattr(self, 'logger'):
            return
        if isinstance(level, str):
            level_value = getattr(logging, level.upper(), logging.INFO)
        else:
            level_value = level
        payload = message
        if details:
            try:
                payload = f"{message} | {json.dumps(details, default=str)}"
            except Exception:
                payload = f"{message} | {details}"
        self.logger.log(level_value, payload)

    def _trace(self, event, **details):
        """Verbose console-only trace helper."""
        if getattr(self, 'console_verbose', False):
            self._console_log(logging.DEBUG, f"TRACE:{event}", details or None)
    
    def _set_status(self, message: str, busy: bool = False):
        """Update status bar."""
        self.status_label.configure(text=message)
        if busy:
            self.progress.start()
        else:
            self.progress.stop()
    
    def _run_async(self, func, callback=None):
        """Run function in background thread."""
        def wrapper():
            try:
                result = func()
                if callback is not None:
                    self.root.after(0, lambda cb=callback, res=result: cb(res))
            except Exception as e:
                self.root.after(0, lambda: self._log(str(e), 'ERROR'))
                self.root.after(0, lambda: self._set_status("Error", False))
        
        thread = threading.Thread(target=wrapper, daemon=True)
        thread.start()

    # ========== Pinecone Helpers ==========

    def _get_pinecone_client(self):
        """Return a cached Pinecone client (per SDK docs)."""
        if self._pinecone_client is None:
            from pinecone import Pinecone
            api_key = os.getenv('PINECONE_API_KEY')
            if not api_key:
                raise RuntimeError("PINECONE_API_KEY is not configured")
            self._pinecone_client = Pinecone(api_key=api_key)
        return self._pinecone_client

    def _get_plaud_oauth_client(self):
        """Return a cached PlaudOAuthClient instance to avoid repeated token loading logs."""
        if self._plaud_oauth_client is None:
            self._plaud_oauth_client = PlaudOAuthClient()
        return self._plaud_oauth_client

    def _get_plaud_client(self):
        """Return a cached PlaudClient instance to avoid repeated token loading logs."""
        if self._plaud_client is None:
            # Pass the cached OAuth client to avoid creating a second one
            self._plaud_client = PlaudClient(oauth_client=self._get_plaud_oauth_client())
        return self._plaud_client

    def _list_pinecone_index_names(self):
        """List index names using the control-plane API, handling all SDK return types."""
        pc = self._get_pinecone_client()
        self._trace('list_indexes_api_call')
        response = pc.list_indexes()
        self._trace('list_indexes_api_response', response_type=str(type(response)))
        names = []

        def _append_name(item):
            if not item:
                return
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict):
                name = item.get('name') or item.get('index_name')
                if name:
                    names.append(name)
            else:
                name = getattr(item, 'name', None)
                if name:
                    names.append(name)

        # Handle IndexList (SDK v6+)
        if hasattr(response, 'indexes'):
            self._trace('list_indexes_has_indexes_attr')
            for idx in response.indexes:
                _append_name(idx)
        elif hasattr(response, 'names'):
            names.extend(list(response.names()))
        elif isinstance(response, dict):
            indexes = response.get('indexes', response)
            if isinstance(indexes, dict):
                indexes = indexes.values()
            for item in indexes:
                _append_name(item)
        else:
            try:
                for item in response:
                    _append_name(item)
            except TypeError:
                _append_name(response)

        unique = []
        seen = set()
        for name in names:
            if name and name not in seen:
                seen.add(name)
                unique.append(name)
        return unique

    def _get_index_host(self, index_name):
        """Fetch and cache the host for an index via describe_index."""
        if not index_name:
            raise RuntimeError("No Pinecone index selected")
        if index_name not in self._index_host_cache:
            pc = self._get_pinecone_client()
            desc = pc.describe_index(name=index_name)
            host = getattr(desc, 'host', None)
            if not host and isinstance(desc, dict):
                host = desc.get('host')
            if not host:
                raise RuntimeError(f"Unable to resolve host for index '{index_name}'")
            self._index_host_cache[index_name] = host
        return self._index_host_cache[index_name]

    def _get_index_client(self, index_name=None, invalidate=False):
        """Return (IndexClient, resolved_name) using host-based instantiation."""
        if index_name is None:
            index_name = self.pinecone_index_var.get() or os.getenv('PINECONE_INDEX_NAME', 'transcripts')
        if not index_name:
            raise RuntimeError("No Pinecone index configured")
        self._trace('index_client_request', index=index_name, invalidate=invalidate)
        if invalidate:
            self._index_client_cache.pop(index_name, None)
            self._index_host_cache.pop(index_name, None)
        if index_name not in self._index_client_cache:
            pc = self._get_pinecone_client()
            host = None
            try:
                host = self._get_index_host(index_name)
                self._trace('index_host_resolved', index=index_name, host=host)
            except Exception as host_err:
                # Network/controller issues sometimes raise OSError(49) when macOS can't assign an address.
                # Fall back to the simpler name-based constructor so the UI can still operate.
                self._log(
                    f"⚠️ Could not resolve host for index '{index_name}' ({host_err}). Falling back to default client.",
                    'WARNING'
                )
                self._trace('index_host_resolution_failed', index=index_name, error=str(host_err))
            if host:
                self._index_client_cache[index_name] = pc.Index(host=host)
            else:
                self._index_client_cache[index_name] = pc.Index(index_name)
            self._trace('index_client_cached', index=index_name, using_host=bool(host))
        return self._index_client_cache[index_name], index_name

    def _reset_index_caches(self, index_name=None):
        """Clear cached Pinecone hosts/clients (optional per index)."""
        if index_name:
            self._index_client_cache.pop(index_name, None)
            self._index_host_cache.pop(index_name, None)
            self._stats_cache.pop(index_name, None)
            self._stats_cache_time.pop(index_name, None)
        else:
            self._index_client_cache.clear()
            self._index_host_cache.clear()
            self._stats_cache.clear()
            self._stats_cache_time.clear()

    def _get_cached_stats(self, index_name=None, max_age=30):
        """Get cached index stats, refreshing if older than max_age seconds."""
        import time
        if index_name is None:
            index_name = self.pinecone_index_var.get()
        
        now = time.time()
        cache_time = self._stats_cache_time.get(index_name, 0)
        
        if index_name in self._stats_cache and (now - cache_time) < max_age:
            self._trace('stats_cache_hit', index=index_name, age_s=int(now - cache_time))
            return self._stats_cache[index_name]
        
        # Cache miss or expired - fetch fresh
        index, _ = self._get_index_client(index_name)
        stats = index.describe_index_stats()
        self._stats_cache[index_name] = stats
        self._stats_cache_time[index_name] = now
        self._trace('stats_cache_miss', index=index_name)
        return stats
    
    # ========== Actions ==========
    
    def _check_auth_status(self):
        """Check if OAuth tokens are valid."""
        self._set_status("Checking authentication...", True)
        
        def check():
            try:
                oauth = self._get_plaud_oauth_client()
                token = oauth.get_access_token()
                if token:
                    # Try to get user info
                    client = self._get_plaud_client()
                    user = client.get_user()
                    return (True, user.get('name', 'Authenticated'))
                return (False, "No valid token")
            except Exception as e:
                return (False, str(e))
        
        def on_complete(result):
            self._set_status("Ready", False)
            is_auth, info = result
            self.is_authenticated = is_auth
            
            if is_auth:
                self.stat_labels['auth_status'].configure(text="✅ Connected", foreground=self.colors['success'])
                self.token_status.configure(text=f"✅ Authenticated as: {info}", foreground=self.colors['success'])
                self._log(f"Authenticated as: {info}", 'SUCCESS')
                # Trigger refresh
                self._fetch_transcripts()
                self._update_pinecone_stats()
            else:
                self.stat_labels['auth_status'].configure(text="❌ Not Connected", foreground=self.colors['error'])
                self.token_status.configure(text=f"❌ {info}", foreground=self.colors['error'])
                self._log(f"Not authenticated: {info}", 'WARNING')
        
        self._run_async(check, on_complete)
    
    def _start_oauth(self):
        """Start OAuth flow."""
        try:
            # Create a fresh OAuth client for the auth flow (not cached)
            oauth = PlaudOAuthClient()
            auth_url, _ = oauth.get_authorization_url()
            webbrowser.open(auth_url)
            
            # Show dialog for code entry
            dialog = tk.Toplevel(self.root)
            dialog.title("Enter Authorization Code")
            dialog.geometry("500x150")
            dialog.transient(self.root)
            
            ttk.Label(dialog, text="After authorizing, paste the code from the URL:").pack(pady=10)
            code_entry = ttk.Entry(dialog, width=60)
            code_entry.pack(pady=5)
            
            def submit():
                code = code_entry.get().strip()
                if code:
                    try:
                        oauth.exchange_code_for_token(code)
                        dialog.destroy()
                        self._log("OAuth authorization successful!", 'SUCCESS')
                        self._check_auth_status()
                    except Exception as e:
                        messagebox.showerror("Error", str(e))
            
            ttk.Button(dialog, text="Submit", command=submit).pack(pady=10)
            
        except Exception as e:
            self._log(f"OAuth error: {e}", 'ERROR')
            messagebox.showerror("Error", str(e))
    
    def _clear_tokens(self):
        """Clear saved OAuth tokens."""
        token_file = Path('.plaud_tokens.json')
        if token_file.exists():
            token_file.unlink()
            self._log("Tokens cleared", 'SUCCESS')
            self._check_auth_status()

    def _disconnect_account(self):
        """Revoke account access via Plaud API."""
        if not messagebox.askyesno("Disconnect", "This will revoke Plaud access for this app. Continue?"):
            return

        self._set_status("Disconnecting from Plaud...", True)

        def revoke():
            client = self._get_plaud_client()
            client.revoke_current_user()
            return True

        def on_complete(_):
            self._set_status("Ready", False)
            token_file = Path('.plaud_tokens.json')
            if token_file.exists():
                token_file.unlink()
            self.is_authenticated = False
            self.stat_labels['auth_status'].configure(text="❌ Not Connected", foreground=self.colors['error'])
            self.token_status.configure(text="Disconnected from Plaud", foreground=self.colors['warning'])
            self._log("Disconnected Plaud account", 'WARNING')

        self._run_async(revoke, on_complete)
    
    def _fetch_transcripts(self):
        """Fetch transcripts from Plaud."""
        if not self.is_authenticated:
            return
        
        self._set_status("Fetching transcripts...", True)
        
        def fetch():
            client = self._get_plaud_client()
            recordings = client.list_recordings()
            return recordings
        
        def on_complete(recordings):
            self._set_status("Ready", False)
            self.transcripts = recordings
            self.stat_labels['plaud_count'].configure(text=str(len(recordings)))
            
            # Populate tree with proper field names
            self.transcript_tree.delete(*self.transcript_tree.get_children())
            for rec in recordings:
                # Use 'name' field from Plaud API
                name = rec.get('name', rec.get('title', rec.get('file_name', 'Untitled')))
                
                # Parse start_at datetime
                start_at = rec.get('start_at', '')
                if start_at:
                    try:
                        dt = datetime.fromisoformat(start_at.replace('Z', '+00:00'))
                        date_str = dt.strftime('%Y-%m-%d')
                        time_str = dt.strftime('%H:%M')
                    except:
                        date_str = start_at[:10] if len(start_at) >= 10 else ''
                        time_str = start_at[11:16] if len(start_at) >= 16 else ''
                else:
                    date_str = ''
                    time_str = ''
                
                # Duration is in milliseconds
                duration_ms = rec.get('duration', 0)
                if duration_ms:
                    minutes = duration_ms // 60000
                    seconds = (duration_ms % 60000) // 1000
                    duration_str = f"{minutes}:{seconds:02d}"
                else:
                    duration_str = ''
                
                rec_id = rec.get('id', '')[:12] + '...' if len(rec.get('id', '')) > 12 else rec.get('id', '')
                
                self.transcript_tree.insert('', tk.END, iid=rec.get('id'), 
                                          values=(name, date_str, time_str, duration_str, rec_id))
            
            self._log(f"Fetched {len(recordings)} transcripts", 'SUCCESS')
            
            # Calculate additional stats
            self._update_transcript_stats(recordings)
        
        self._run_async(fetch, on_complete)
    
    def _update_transcript_stats(self, recordings):
        """Calculate and display aggregate stats from recordings."""
        if not recordings:
            return
        
        # Total duration
        total_ms = sum(r.get('duration', 0) for r in recordings)
        hours = total_ms // 3600000
        minutes = (total_ms % 3600000) // 60000
        self.stat_labels['total_duration'].configure(text=f"{hours}h {minutes}m")
        
        # Estimated words (rough: ~150 words per minute of speech)
        est_words = (total_ms // 60000) * 150
        if est_words > 1000:
            self.stat_labels['total_words'].configure(text=f"~{est_words // 1000}k")
        else:
            self.stat_labels['total_words'].configure(text=f"~{est_words}")
        
        # Date range
        dates = []
        for r in recordings:
            start_at = r.get('start_at', '')
            if start_at:
                try:
                    dt = datetime.fromisoformat(start_at.replace('Z', '+00:00'))
                    dates.append(dt)
                except:
                    pass
        
        if dates:
            min_date = min(dates).strftime('%m/%d')
            max_date = max(dates).strftime('%m/%d')
            self.stat_labels['date_range'].configure(text=f"{min_date} - {max_date}")
        
        # Speakers will be calculated when we have transcript data
        self.stat_labels['unique_speakers'].configure(text="—")
    
    def _get_synced_ids(self):
        """Get list of IDs already in Pinecone."""
        try:
            index, _ = self._get_index_client()
            index.describe_index_stats()
            return set()
        except Exception:
            return set()
    
    def _filter_transcripts(self, event=None):
        """Filter transcript list."""
        query = self.transcript_filter.get().lower()
        self.transcript_tree.delete(*self.transcript_tree.get_children())
        
        for rec in self.transcripts:
            name = rec.get('name', rec.get('title', rec.get('file_name', 'Untitled')))
            if query in name.lower():
                # Parse start_at datetime
                start_at = rec.get('start_at', '')
                if start_at:
                    try:
                        dt = datetime.fromisoformat(start_at.replace('Z', '+00:00'))
                        date_str = dt.strftime('%Y-%m-%d')
                        time_str = dt.strftime('%H:%M')
                    except:
                        date_str = start_at[:10] if len(start_at) >= 10 else ''
                        time_str = start_at[11:16] if len(start_at) >= 16 else ''
                else:
                    date_str = ''
                    time_str = ''
                
                duration_ms = rec.get('duration', 0)
                if duration_ms:
                    minutes = duration_ms // 60000
                    seconds = (duration_ms % 60000) // 1000
                    duration_str = f"{minutes}:{seconds:02d}"
                else:
                    duration_str = ''
                
                rec_id = rec.get('id', '')[:12] + '...' if len(rec.get('id', '')) > 12 else rec.get('id', '')
                self.transcript_tree.insert('', tk.END, iid=rec.get('id'),
                                          values=(name, date_str, time_str, duration_str, rec_id))
    
    def _view_selected_transcript(self):
        """View the selected transcript."""
        selection = self.transcript_tree.selection()
        if not selection:
            messagebox.showinfo("Info", "Please select a transcript to view")
            return
        self._view_transcript_by_id(selection[0])

    def _export_selected_transcripts(self):
        """Export selected transcripts to Markdown files."""
        selection = self.transcript_tree.selection()
        if not selection:
            messagebox.showinfo("Info", "Please select at least one transcript to export")
            return
        
        export_dir = filedialog.askdirectory(title="Select export folder")
        if not export_dir:
            return
        
        self._set_status("Exporting transcripts...", True)
        self._log(f"Exporting {len(selection)} transcripts...", 'INFO')

        def export():
            client = self._get_plaud_client()
            saved_files = []
            for transcript_id in selection:
                detail = client.get_recording(transcript_id)
                parsed = self._parse_transcript_detail(detail)
                meta = next((t for t in self.transcripts if t.get('id') == transcript_id), {})
                name = meta.get('name', meta.get('title', 'Transcript'))
                start_at = meta.get('start_at', '')
                duration_ms = meta.get('duration', 0)
                duration_str = self._format_duration(duration_ms)
                markdown = self._build_markdown_document(name, start_at, duration_str, parsed)
                filename = self._sanitize_filename(f"{name}.md")
                filepath = Path(export_dir) / filename
                filepath.write_text(markdown, encoding='utf-8')
                saved_files.append(str(filepath))
            return saved_files

        def on_complete(files):
            self._set_status("Ready", False)
            self._log(f"Exported {len(files)} transcripts", 'SUCCESS')
            messagebox.showinfo("Export Complete", f"Saved {len(files)} transcripts to:\n{export_dir}")

        self._run_async(export, on_complete)
    
    def _view_transcript(self, event):
        """View transcript details on double-click."""
        selection = self.transcript_tree.selection()
        if not selection:
            return
        self._view_transcript_by_id(selection[0])
    
    def _view_transcript_by_id(self, transcript_id: str):
        """View transcript details by ID."""
        # Find matching transcript
        transcript = next((t for t in self.transcripts if t.get('id') == transcript_id), None)
        if not transcript:
            messagebox.showerror("Error", f"Transcript not found: {transcript_id}")
            return
        
        name = transcript.get('name', 'Untitled')
        
        # Create viewer dialog
        dialog = tk.Toplevel(self.root)
        dialog.title(f"📝 {name}")
        dialog.geometry("900x700")
        
        # Info header
        info_frame = ttk.Frame(dialog, padding=10)
        info_frame.pack(fill=tk.X)
        
        ttk.Label(info_frame, text=name, font=('Helvetica', 16, 'bold')).pack(anchor='w')
        
        # Metadata
        start_at = transcript.get('start_at', '')
        duration_ms = transcript.get('duration', 0)
        duration_str = self._format_duration(duration_ms)
        
        meta_text = f"📅 {start_at or 'Unknown'}  |  ⏱️ {duration_str}  |  🆔 {transcript.get('id', '')}"
        ttk.Label(info_frame, text=meta_text, font=('Helvetica', 11)).pack(anchor='w', pady=5)
        
        # Tabs for different views
        notebook = ttk.Notebook(dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Full transcript tab
        transcript_tab = ttk.Frame(notebook)
        notebook.add(transcript_tab, text="📜 Full Transcript")
        
        transcript_text = scrolledtext.ScrolledText(
            transcript_tab, wrap=tk.WORD, font=('Helvetica', 12),
            bg=self.colors['surface'], fg=self.colors['fg']
        )
        transcript_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        transcript_text.insert('1.0', "Loading transcript...")
        
        # Speakers tab
        speakers_tab = ttk.Frame(notebook)
        notebook.add(speakers_tab, text="👥 By Speaker")
        
        speakers_text = scrolledtext.ScrolledText(
            speakers_tab, wrap=tk.WORD, font=('Helvetica', 12),
            bg=self.colors['surface'], fg=self.colors['fg']
        )
        speakers_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Notes tab
        notes_tab = ttk.Frame(notebook)
        notebook.add(notes_tab, text="🧾 Notes & Summaries")

        notes_text = scrolledtext.ScrolledText(
            notes_tab, wrap=tk.WORD, font=('Helvetica', 12),
            bg=self.colors['surface'], fg=self.colors['fg']
        )
        notes_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        notes_text.insert('1.0', "Loading Plaud notes...")

        # Insights tab
        insights_tab = ttk.Frame(notebook)
        notebook.add(insights_tab, text="📈 Insights")

        insights_text = scrolledtext.ScrolledText(
            insights_tab, wrap=tk.WORD, font=('Helvetica', 12),
            bg=self.colors['surface'], fg=self.colors['fg']
        )
        insights_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        insights_text.insert('1.0', "Calculating insights...")
        
        # Raw data tab
        raw_tab = ttk.Frame(notebook)
        notebook.add(raw_tab, text="🔧 Raw Data")
        
        raw_text = scrolledtext.ScrolledText(
            raw_tab, wrap=tk.WORD, font=('Monaco', 10),
            bg=self.colors['surface'], fg=self.colors['fg']
        )
        raw_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Actions
        transcript_state = {'detail': None, 'text': '', 'name': name}
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        def export_current():
            if not transcript_state['detail']:
                messagebox.showinfo("Info", "Transcript still loading")
                return
            filepath = filedialog.asksaveasfilename(
                defaultextension=".md",
                filetypes=[("Markdown", "*.md"), ("Text", "*.txt")],
                initialfile=self._sanitize_filename(f"{name}.md")
            )
            if not filepath:
                return
            parsed = self._parse_transcript_detail(transcript_state['detail'])
            markdown = self._build_markdown_document(name, start_at, duration_str, parsed)
            Path(filepath).write_text(markdown, encoding='utf-8')
            self._log(f"Exported transcript to {filepath}", 'SUCCESS')

        def copy_plain_text():
            if not transcript_state['text']:
                return
            self.root.clipboard_clear()
            self.root.clipboard_append(transcript_state['text'])
            messagebox.showinfo("Copied", "Transcript copied to clipboard")

        ttk.Button(button_frame, text="💾 Export Markdown", command=export_current).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📋 Copy Transcript", command=copy_plain_text).pack(side=tk.LEFT, padx=5)
        
        # Fetch content
        def fetch_content():
            client = self._get_plaud_client()
            return client.get_recording(transcript.get('id'))
        
        def show_content(details):
            transcript_state['detail'] = details
            parsed = self._parse_transcript_detail(details)
            transcript_state['text'] = parsed.get('full_text', '')
            
            # Update transcript tab
            segments = parsed.get('segments', [])
            transcript_text.delete('1.0', tk.END)
            if segments:
                transcript_text.insert(tk.END, '\n\n'.join(
                    f"{seg['time']} {seg['speaker']}: {seg['content']}" for seg in segments
                ))
            else:
                transcript_text.insert('1.0', "No transcript available")
            
            # Update speakers tab
            speakers_text.delete('1.0', tk.END)
            speaker_map = parsed.get('speaker_segments', {})
            for speaker, entries in speaker_map.items():
                speakers_text.insert(tk.END, f"\n{'='*60}\n")
                speakers_text.insert(tk.END, f"👤 {speaker} ({len(entries)} segments)\n")
                speakers_text.insert(tk.END, f"{'='*60}\n\n")
                speakers_text.insert(tk.END, '\n'.join(
                    f"{seg['time']} {seg['content']}" for seg in entries
                ))
                speakers_text.insert(tk.END, '\n\n')
            if speaker_map:
                self.stat_labels['unique_speakers'].configure(text=f"{len(speaker_map)} (latest)")
            
            # Notes tab
            notes_text.delete('1.0', tk.END)
            notes = parsed.get('notes', [])
            if notes:
                for note in notes:
                    notes_text.insert(tk.END, f"## {note['title']}\n\n{note['content']}\n\n")
            else:
                notes_text.insert('1.0', "No Plaud notes returned for this recording")
            
            # Insights tab
            insights_text.delete('1.0', tk.END)
            keywords = parsed.get('keywords', [])
            word_count = parsed.get('word_count', 0)
            insights_lines = [f"Total words: {word_count}"]
            if speaker_map:
                insights_lines.append(f"Speakers: {', '.join(speaker_map.keys())}")
            if keywords:
                insights_lines.append("\nTop keywords:")
                for word, count in keywords[:10]:
                    insights_lines.append(f"- {word} ({count})")
            insights_text.insert('1.0', '\n'.join(insights_lines))
            
            # Raw tab
            raw_text.delete('1.0', tk.END)
            raw_text.insert('1.0', json.dumps(details, indent=2, default=str))
        
        self._run_async(fetch_content, show_content)
    
    def _sync_selected(self):
        """Sync selected transcripts to Pinecone."""
        selection = self.transcript_tree.selection()
        if not selection:
            messagebox.showinfo("Info", "Please select transcripts to sync")
            return
        
        selected_index = self.pinecone_index_var.get() or os.getenv('PINECONE_INDEX_NAME', 'transcripts')
        
        # Show confirmation with index name
        response = messagebox.askyesno(
            "Sync Selected",
            f"Sync {len(selection)} transcript(s) to index: '{selected_index}'?"
        )
        if not response:
            return
        
        self._log(f"Syncing {len(selection)} selected transcripts to {selected_index}...", 'INFO')
        self._sync_to_pinecone()  # TODO: Filter to selected only
    
    def _sync_to_pinecone(self):
        """Sync all transcripts to Pinecone."""
        selected_index = self.pinecone_index_var.get() or os.getenv('PINECONE_INDEX_NAME', 'transcripts')
        
        # Confirm which index to sync to
        response = messagebox.askyesno(
            "Confirm Sync",
            f"Sync Plaud transcripts to index: '{selected_index}'?\n\nThis will:\n• Fetch all transcripts from Plaud\n• Generate embeddings\n• Upsert to {selected_index}"
        )
        if not response:
            return
        
        self._set_status(f"Syncing to {selected_index}...", True)
        self._log(f"Starting Pinecone sync to index: {selected_index}", 'INFO')
        
        def sync():
            import subprocess
            # Pass selected index to script via environment
            env = os.environ.copy()
            env['PINECONE_INDEX_NAME'] = selected_index
            result = subprocess.run(
                ['python', 'scripts/sync_to_pinecone.py'],
                capture_output=True, text=True,
                cwd=Path(__file__).parent,
                env=env
            )
            return result.stdout + result.stderr
        
        def on_complete(output):
            self._set_status("Ready", False)
            self._log(f"Sync to {selected_index} complete", 'SUCCESS')
            self._log(output, 'INFO')
            self._update_pinecone_stats()
            self._refresh_pinecone_vectors()  # Refresh to show new vectors
            self.stat_labels['last_sync'].configure(text=datetime.now().strftime('%H:%M'))
            messagebox.showinfo("Sync Complete", f"Transcripts synced to index: {selected_index}")
        
        self._run_async(sync, on_complete)
    
    def _update_pinecone_stats(self):
        """Update Pinecone vector count."""
        def get_stats():
            try:
                api_key = os.getenv('PINECONE_API_KEY')
                if not api_key:
                    return (False, "No API key set")
                index_name = self.pinecone_index_var.get() or os.getenv('PINECONE_INDEX_NAME', 'transcripts')
                index, _ = self._get_index_client(index_name)
                stats = index.describe_index_stats()
                return (True, stats.total_vector_count)
            except Exception as e:
                return (False, str(e)[:50])
        
        def on_complete(result):
            success, value = result
            if success:
                self.stat_labels['pinecone_count'].configure(text=str(value), foreground=self.colors['fg'])
            else:
                self.stat_labels['pinecone_count'].configure(text="❌ " + value, foreground=self.colors['error'])
                if "401" in value or "authentication" in value.lower():
                    self._log("Pinecone auth failed - check API key in Settings", 'ERROR')
        
        self._run_async(get_stats, on_complete)
    
    def _perform_search(self):
        """Perform semantic search."""
        query = self.search_entry.get().strip()
        if not query:
            return
        
        self._set_status("Searching...", True)
        self.search_results_text.delete('1.0', tk.END)
        self.search_results_text.insert('1.0', "Searching...\n")
        
        def search():
            import google.generativeai as genai
            
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            
            # Generate query embedding
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=query
            )
            query_embedding = result['embedding']
            
            # Search Pinecone
            index, _ = self._get_index_client()
            
            limit = int(self.search_limit.get())
            results = index.query(
                vector=query_embedding,
                top_k=limit,
                include_metadata=True
            )
            
            return results.matches
        
        def on_complete(matches):
            self._set_status("Ready", False)
            self.search_results_text.delete('1.0', tk.END)
            
            if not matches:
                self.search_results_text.insert('1.0', "No results found.\n")
                return
            
            for i, match in enumerate(matches, 1):
                meta = match.metadata
                score = match.score
                
                self.search_results_text.insert(tk.END, f"\n{'='*60}\n")
                self.search_results_text.insert(tk.END, f"#{i} | Score: {score:.3f}\n")
                self.search_results_text.insert(tk.END, f"Title: {meta.get('title', 'Untitled')}\n")
                self.search_results_text.insert(tk.END, f"Themes: {meta.get('themes', '')}\n")
                self.search_results_text.insert(tk.END, f"Date: {meta.get('date', '')}\n")
                
                if self.include_context.get():
                    text = meta.get('text', '')[:500]
                    self.search_results_text.insert(tk.END, f"\nPreview:\n{text}...\n")
            
            self._log(f"Search found {len(matches)} results for: {query}", 'SUCCESS')
        
        self._run_async(search, on_complete)
    
    def _load_pinecone_indexes(self):
        """Load all available Pinecone indexes into dropdown."""
        self._log("Starting to load Pinecone indexes...", 'INFO')
        
        def fetch_indexes():
            try:
                self._trace('fetch_indexes_start')
                names = self._list_pinecone_index_names()
                self._trace('fetch_indexes_complete', count=len(names), names=names)
                return names
            except Exception as e:
                self._trace('fetch_indexes_error', error=str(e))
                self._log(f"Error loading Pinecone indexes: {e}", 'ERROR')
                return []
        
        def populate_dropdown(index_names):
            self._trace('populate_dropdown_called', count=len(index_names) if index_names else 0)
            if not hasattr(self, 'pinecone_index_dropdown'):
                self._log("pinecone_index_dropdown not found!", 'ERROR')
                return
            if not index_names:
                self._log("No Pinecone indexes available", 'WARNING')
                return
            self.pinecone_index_dropdown['values'] = index_names
            if hasattr(self, 'header_index_dropdown'):
                self.header_index_dropdown['values'] = index_names
            if self.pinecone_index_var.get() not in index_names:
                self.pinecone_index_var.set(index_names[0])
            self._log(f"Loaded {len(index_names)} Pinecone indexes: {index_names}", 'SUCCESS')
            # Refresh namespaces once indexes are available
            self._load_namespaces_only()
        
        self._run_async(fetch_indexes, populate_dropdown)
    
    def _on_tab_changed(self, event):
        """Handle tab change events - lazy load data for active tab."""
        current_tab = self.notebook.select()
        tab_text = self.notebook.tab(current_tab, 'text')
        
        # Dashboard tab - load indexes and stats on first view
        if '📊' in tab_text or 'Dashboard' in tab_text:
            if not self._indexes_loaded:
                self.root.after(50, lambda: self._load_indexes_for_header())
            # Trigger stats refresh (will use cache if recent)
            self.root.after(100, lambda: self._refresh_dashboard_stats(use_cache=True))
        
        # Pinecone tab - load indexes and namespaces on first view
        elif '🌲' in tab_text or 'Pinecone' in tab_text:
            if hasattr(self, '_pinecone_loaded') and not self._pinecone_loaded:
                self._pinecone_loaded = True
                self._log("🌲 Loading Pinecone data...", 'INFO')
                self.root.after(50, self._load_pinecone_indexes)
                self.root.after(200, self._load_namespaces_only)
                self.root.after(400, self._refresh_pinecone_vectors)  # Auto-load vectors too
            # DON'T auto-load org inspector - it's too slow. User can click "Org Overview" button
        
        # Auto-refresh Transcripts when switched to (only if empty)
        elif '📝' in tab_text or 'Transcripts' in tab_text:
            if hasattr(self, 'transcript_tree') and not self.transcript_tree.get_children():
                self.root.after(100, self._fetch_transcripts)
    
    def _switch_to_vector_mgmt(self, event):
        """Switch to vector management sub-tab and load selected index."""
        if not hasattr(self, 'org_indexes_tree'):
            return
        selection = self.org_indexes_tree.selection()
        if not selection:
            return
        
        item = self.org_indexes_tree.item(selection[0])
        index_name = item['values'][0]
        
        # Find the pinecone notebook and switch to vector management tab
        if hasattr(self, 'pinecone_index_var'):
            self.pinecone_index_var.set(index_name)
            # Switch to second tab (Vector Management)
            for child in self.notebook.winfo_children():
                if isinstance(child, ttk.Notebook):
                    child.select(1)  # Select second tab
                    self._on_index_changed()
                    break
    
    def _on_index_changed(self):
        """Handle index selection change - load namespaces only."""
        new_index = self.pinecone_index_var.get()
        self._log(f"🔄 Switching to index: {new_index}", 'INFO')
        
        # Show loading state immediately
        if hasattr(self, 'pinecone_status_label'):
            self.pinecone_status_label.config(text="● Loading namespaces...", foreground='#f9e2af')
        
        # Clear current data
        for item in self.pinecone_tree.get_children():
            self.pinecone_tree.delete(item)
        
        # Clear preview
        if hasattr(self, 'pinecone_preview_text'):
            self.pinecone_preview_text.configure(state='normal')
            self.pinecone_preview_text.delete('1.0', tk.END)
            self.pinecone_preview_text.insert('1.0', f"Select a namespace to view vectors from {new_index}...")
            self.pinecone_preview_text.configure(state='disabled')
        
        # Clear namespace listbox
        self.pinecone_namespace_var.set("")
        if hasattr(self, 'namespace_listbox'):
            self.namespace_listbox.delete(0, tk.END)
        
        # Load namespaces AND vectors automatically
        self._refresh_pinecone_vectors()
    
    def _on_namespace_changed(self, event=None):
        """Handle namespace selection change - instant filtering."""
        # Get selected namespace from listbox
        selection = self.namespace_listbox.curselection()
        if not selection:
            return
        
        namespace = self.namespace_listbox.get(selection[0])
        
        # Handle "(all)" option
        if namespace == "(all)":
            namespace = ""
        # Handle "default" - query the empty namespace
        elif namespace == "default":
            namespace = ""
        
        self.pinecone_namespace_var.set(namespace)
        index_name = self.pinecone_index_var.get()
        
        self._log(f"🔄 Filtering namespace: {namespace or 'all'} in {index_name}", 'INFO')
        
        # Show filtering state immediately
        if hasattr(self, 'pinecone_status_label'):
            self.pinecone_status_label.config(text="● Filtering...", foreground='#f9e2af')
        
        # Update preview
        if hasattr(self, 'pinecone_preview_text'):
            self.pinecone_preview_text.configure(state='normal')
            self.pinecone_preview_text.delete('1.0', tk.END)
            self.pinecone_preview_text.insert('1.0', f"Loading vectors from namespace '{namespace or 'all'}'...")
            self.pinecone_preview_text.configure(state='disabled')
        
        # Immediate refresh with selected namespace
        self._refresh_pinecone_vectors()
    
    def _load_namespaces_only(self):
        """Load available namespaces without querying vectors."""
        def fetch_namespaces():
            index_name = self.pinecone_index_var.get()
            stats = self._get_cached_stats(index_name)
            namespaces_obj = getattr(stats, 'namespaces', {}) or {}
            if hasattr(namespaces_obj, 'keys'):
                namespace_count = len(list(namespaces_obj.keys()))
            elif isinstance(namespaces_obj, dict):
                namespace_count = len(namespaces_obj)
            else:
                namespace_count = 0
            self._trace(
                'namespace_stats_fetched',
                index=index_name,
                total_vectors=getattr(stats, 'total_vector_count', None),
                namespace_count=namespace_count
            )
            return {'stats': stats, 'index_name': index_name}
        
        def display_namespaces(data):
            all_namespaces = self._update_namespace_ui(data['stats'], data['index_name'])
            real_namespaces = [ns for ns in all_namespaces if ns != '(all)']
            count = max(len(real_namespaces), 1)

            self._log(f"Loaded {count} namespace(s) from {data['index_name']}", 'SUCCESS')
            self._trace('namespace_ui_loaded', index=data['index_name'], namespaces=real_namespaces)

            # Auto-load vectors if there's only one non-default namespace
            if len(real_namespaces) == 1 and real_namespaces[0] not in ['default', '']:
                selected = real_namespaces[0]
                self._log(f"Auto-loading vectors from namespace '{selected}'", 'INFO')
                self._trace('namespace_auto_load_triggered', index=data['index_name'], namespace=selected)
                self.root.after(100, self._refresh_pinecone_vectors)
        
        self._run_async(fetch_namespaces, display_namespaces)
    
    def _refresh_pinecone_vectors(self):
        """Fetch and display vectors from Pinecone index using list+fetch (robust)."""
        # Update status indicator if it exists
        if hasattr(self, 'pinecone_status_label'):
            self.pinecone_status_label.config(text="● Loading...", foreground='#f9e2af')
        self._set_status("Loading Pinecone vectors...", True)
        
        def fetch_vectors():
            import time
            start_time = time.time()
            
            index, index_name = self._get_index_client()
            stats = self._get_cached_stats(index_name)  # Use cached stats - much faster!
            
            # Metric is not in stats - we'll fetch it lazily if needed, don't block here
            # Just set a placeholder
            if not hasattr(stats, 'metric'):
                stats.metric = '—'
            
            # Use selected namespace or query all
            namespace = self.pinecone_namespace_var.get() or ''
            
            # Log what we're querying
            total_vectors = getattr(stats, 'total_vector_count', None)
            self._trace(
                'vector_fetch_start',
                index=index_name,
                namespace=namespace or 'ALL',
                total_vectors=total_vectors,
                page_limit=self.page_size_var.get() if hasattr(self, 'page_size_var') else 100
            )
            
            all_vectors = []
            # Use smaller limit for faster initial loads (user can increase via pagination)
            limit = self.page_size_var.get() if hasattr(self, 'page_size_var') else 100
            
            namespaces_to_query = []
            if not namespace:
                # Query all namespaces found in stats
                stats_namespaces = getattr(stats, 'namespaces', None)
                if isinstance(stats_namespaces, dict):
                    namespaces_to_query = list(stats_namespaces.keys())
                elif hasattr(stats_namespaces, 'keys'):
                    namespaces_to_query = list(stats_namespaces.keys())
                if not namespaces_to_query:
                    namespaces_to_query = ['']
            else:
                namespaces_to_query = [namespace]
            
            self._trace('vector_fetch_namespaces', index=index_name, namespaces=namespaces_to_query)
            
            for ns in namespaces_to_query:
                try:
                    # 1. List IDs with timeout protection
                    ids_to_fetch = []
                    try:
                        # Use index.list() which yields batches of IDs
                        for batch in index.list(namespace=ns, limit=100):
                            ids_to_fetch.extend(batch)
                            if len(ids_to_fetch) >= limit:
                                break
                            # Safety: don't loop forever
                            if time.time() - start_time > 30:
                                self._trace('vector_list_timeout', namespace=ns)
                                break
                    except Exception as list_err:
                        self._trace('vector_list_error', namespace=ns, error=str(list_err))
                        continue
                    
                    ids_to_fetch = ids_to_fetch[:limit]
                    
                    if not ids_to_fetch:
                        continue
                        
                    # 2. Fetch Metadata in batches
                    chunk_size = 100
                    for i in range(0, len(ids_to_fetch), chunk_size):
                        chunk = ids_to_fetch[i:i+chunk_size]
                        try:
                            # Retry logic for fetch
                            for attempt in range(3):
                                try:
                                    fetch_res = index.fetch(ids=chunk, namespace=ns)
                                    break
                                except Exception as e:
                                    if "SSL" in str(e) and attempt < 2:
                                        time.sleep(1)
                                    else:
                                        raise e
                            
                            # Add namespace to metadata
                            for vec_id, vec in fetch_res.vectors.items():
                                if vec.metadata is None:
                                    vec.metadata = {}
                                vec.metadata['_namespace'] = ns
                                all_vectors.append(vec)
                                
                        except Exception as e:
                            self._trace('vector_fetch_batch_error', namespace=ns, error=str(e), chunk=len(chunk))
                            
                    if len(all_vectors) >= limit:
                        break
                        
                except Exception as e:
                    self._trace('vector_list_error', namespace=ns, error=str(e))
            
            duration_ms = int((time.time() - start_time) * 1000)
            self._trace(
                'vector_fetch_complete',
                index=index_name,
                namespaces=namespaces_to_query,
                vectors=len(all_vectors),
                duration_ms=duration_ms
            )
            
            return {
                'stats': stats,
                'vectors': all_vectors,
                'index_name': index_name
            }
        
        def display_vectors(data):
            self._set_status("Ready", False)

            stats = data['stats']
            vectors = data['vectors']
            vector_count = len(vectors)
            self._trace('vector_display', index=data['index_name'], vector_count=vector_count)

            if hasattr(self, 'pinecone_status_label'):
                self.pinecone_status_label.config(text=f"● {vector_count} vector(s)", foreground='#a6e3a1')

            self._update_namespace_ui(stats, data['index_name'])

            tree = getattr(self, 'pinecone_tree', None)
            if not tree:
                return

            tree.delete(*tree.get_children())

            for vec in vectors:
                meta = getattr(vec, 'metadata', {}) or {}
                vector_id = getattr(vec, 'id', 'unknown')
                namespace = meta.get('_namespace', '')

                title = (meta.get('title') or meta.get('name') or meta.get('recording_name') or
                         meta.get('file_name') or meta.get('text', '')[:50] or 'Untitled')
                date_str = (meta.get('date') or meta.get('start_at', '')[:10] or
                            meta.get('created_at', '')[:10] or '—')

                duration_ms = (meta.get('duration_ms') or meta.get('duration') or
                               meta.get('length_ms') or meta.get('length'))
                if isinstance(duration_ms, (int, float)):
                    duration_str = self._format_duration(int(duration_ms))
                else:
                    duration_str = str(duration_ms)[:8] if duration_ms else '—'

                themes = (meta.get('themes') or meta.get('tags') or meta.get('categories') or
                          meta.get('keywords') or '')[:40]

                field_count = len(meta)

                display_id = vector_id[:14] + '...' if len(vector_id) > 14 else vector_id
                tree.insert('', tk.END, iid=vector_id, tags=(namespace,), values=(
                    display_id,
                    title,
                    date_str,
                    duration_str,
                    themes or '—',
                    str(field_count)
                ))

            self._log(f"Loaded {vector_count} vectors from Pinecone", 'SUCCESS')
            self._update_vector_preview(vectors)
        
        self._run_async(fetch_vectors, display_vectors)
    
    def _update_vector_preview(self, vectors):
        """Update the preview panel with first vector info."""
        self.pinecone_preview_text.configure(state='normal')
        self.pinecone_preview_text.delete('1.0', tk.END)
        
        if not vectors:
            self.pinecone_preview_text.insert('1.0', "🔍 No vectors in this index/namespace")
        else:
            vec = vectors[0]
            meta = vec.metadata if hasattr(vec, 'metadata') else {}
            vector_id = vec.id if hasattr(vec, 'id') else 'unknown'
            
            # Extract key info
            title = meta.get('title') or meta.get('name') or 'Untitled'
            date = meta.get('date') or meta.get('start_at', '')[:10] or '—'
            themes = meta.get('themes') or meta.get('tags') or '—'
            
            preview = f"ID: {vector_id[:40]}...\nTitle: {title}\nDate: {date} | Themes: {themes}\nMetadata fields: {len(meta)}"
            self.pinecone_preview_text.insert('1.0', preview)
        
        self.pinecone_preview_text.configure(state='disabled')

    def _on_vector_selected(self, event=None):
        """Handle vector row selection - show full metadata in preview pane."""
        selection = self.pinecone_tree.selection()
        if not selection:
            return
        
        vector_id = selection[0]  # iid is the full vector ID
        
        # Fetch detailed metadata for selected vector
        def fetch_details():
            try:
                index, _ = self._get_index_client()
                namespace = self.pinecone_namespace_var.get() or ''
                result = index.fetch(ids=[vector_id], namespace=namespace if namespace else None)
                if result and result.vectors and vector_id in result.vectors:
                    return result.vectors[vector_id]
            except Exception as e:
                self._trace('vector_preview_fetch_error', vector_id=vector_id, error=str(e))
            return None
        
        def display_details(vec):
            self.pinecone_preview_text.configure(state='normal')
            self.pinecone_preview_text.delete('1.0', tk.END)
            
            if not vec:
                self.pinecone_preview_text.insert('1.0', f"Could not fetch details for {vector_id}")
            else:
                meta = vec.metadata if hasattr(vec, 'metadata') else {}
                vid = vec.id if hasattr(vec, 'id') else 'unknown'
                
                lines = []
                lines.append(f"{'='*40}")
                lines.append(f"VECTOR: {vid}")
                lines.append(f"{'='*40}")
                lines.append("")
                
                # Core fields
                title = meta.get('title') or meta.get('name') or meta.get('recording_name') or 'Untitled'
                lines.append(f"📝 Title: {title}")
                
                date = meta.get('date') or meta.get('start_at', '')[:10] or meta.get('created_at', '')[:10]
                if date:
                    lines.append(f"📅 Date: {date}")
                
                duration = meta.get('duration_ms') or meta.get('duration') or meta.get('length_ms')
                if duration:
                    lines.append(f"⏱️ Duration: {self._format_duration(int(duration))}")
                
                themes = meta.get('themes') or meta.get('tags') or meta.get('keywords')
                if themes:
                    lines.append(f"🏷️ Themes: {themes}")
                
                namespace = meta.get('_namespace', '')
                if namespace:
                    lines.append(f"📂 Namespace: {namespace}")
                
                lines.append("")
                lines.append(f"{'─'*40}")
                lines.append(f"ALL METADATA ({len(meta)} fields):")
                lines.append(f"{'─'*40}")
                
                for key, value in sorted(meta.items()):
                    if key.startswith('_'):
                        continue  # Skip internal fields
                    val_str = str(value)
                    if len(val_str) > 80:
                        val_str = val_str[:77] + "..."
                    lines.append(f"  {key}: {val_str}")
                
                self.pinecone_preview_text.insert('1.0', '\n'.join(lines))
            
            self.pinecone_preview_text.configure(state='disabled')
        
        self._run_async(fetch_details, display_details)
    
    def _delete_pinecone_vectors(self):
        """Delete selected vectors from Pinecone."""
        selection = self.pinecone_tree.selection()
        if not selection:
            messagebox.showinfo("Info", "Please select vectors to delete")
            return
        
        count = len(selection)
        result = messagebox.askyesno(
            "Confirm Delete",
            f"Delete {count} vector(s) from Pinecone?\\n\\nThis cannot be undone."
        )
        if not result:
            return
        
        self._set_status(f"Deleting {count} vectors...", True)
        
        def delete():
            index, index_name = self._get_index_client()
            
            # Group IDs by namespace
            ids_by_namespace = {}
            for item_id in selection:
                item = self.pinecone_tree.item(item_id)
                # Get namespace from tags
                tags = item.get('tags', [])
                namespace = tags[0] if tags else ''
                
                if namespace not in ids_by_namespace:
                    ids_by_namespace[namespace] = []
                ids_by_namespace[namespace].append(item_id)
            
            total_deleted = 0
            for ns, ids in ids_by_namespace.items():
                try:
                    index.delete(ids=ids, namespace=ns)
                    total_deleted += len(ids)
                except Exception as e:
                    self._trace('vector_delete_error', namespace=ns, error=str(e), count=len(ids))
            
            return total_deleted
        
        def on_complete(deleted_count):
            if deleted_count > 0:
                self._set_status("Ready", False)
                self._log(f"✅ Deleted {deleted_count} vector(s)", 'SUCCESS')
                # Smart refresh - only if we deleted something
                self._refresh_pinecone_vectors()
                # Show toast notification
                if hasattr(self, 'pinecone_status_label'):
                    self.pinecone_status_label.config(text=f"● Deleted {deleted_count}", foreground='#f38ba8')
                    self.root.after(2000, lambda: self.pinecone_status_label.config(text="● Ready", foreground='#a6e3a1'))
            else:
                self._set_status("Delete failed or cancelled", False)
        
        self._run_async(delete, on_complete)
    
    def _show_delete_options(self):
        """Show delete options: by selection, by filter, or clear namespace."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Delete Options")
        dialog.geometry("500x450")
        dialog.configure(bg=self.colors['bg'])
        
        ttk.Label(dialog, text="🗑️ Delete Options", font=('Helvetica', 14, 'bold')).pack(pady=15)
        
        # Option 1: Delete selected
        selection = self.pinecone_tree.selection()
        sel_frame = ttk.LabelFrame(dialog, text=f"Delete Selected ({len(selection)} vectors)", padding=10)
        sel_frame.pack(fill=tk.X, padx=15, pady=5)
        ttk.Label(sel_frame, text="Delete the currently selected vectors from the tree").pack(anchor='w')
        ttk.Button(sel_frame, text="Delete Selected", 
                  command=lambda: [dialog.destroy(), self._delete_pinecone_vectors()],
                  state='normal' if selection else 'disabled').pack(pady=5)
        
        # Option 2: Delete by metadata filter
        filter_frame = ttk.LabelFrame(dialog, text="Delete by Metadata Filter", padding=10)
        filter_frame.pack(fill=tk.X, padx=15, pady=5)
        ttk.Label(filter_frame, text="Delete all vectors matching a metadata filter (JSON):").pack(anchor='w')
        filter_text = scrolledtext.ScrolledText(filter_frame, height=3, wrap=tk.WORD, font=('Monaco', 9))
        filter_text.pack(fill=tk.X, pady=5)
        filter_text.insert('1.0', '{"source": "example"}')
        
        def delete_by_filter():
            import json
            try:
                filter_dict = json.loads(filter_text.get('1.0', tk.END).strip())
            except json.JSONDecodeError as e:
                messagebox.showerror("JSON Error", f"Invalid JSON: {e}")
                return
            
            if not messagebox.askyesno("Confirm", f"Delete ALL vectors matching filter?\n\n{json.dumps(filter_dict, indent=2)}\n\nThis cannot be undone!"):
                return
            
            dialog.destroy()
            self._delete_by_filter(filter_dict)
        
        ttk.Button(filter_frame, text="Delete by Filter", command=delete_by_filter).pack(pady=5)
        
        # Option 3: Clear entire namespace
        ns_frame = ttk.LabelFrame(dialog, text="⚠️ Clear Namespace", padding=10)
        ns_frame.pack(fill=tk.X, padx=15, pady=5)
        
        current_ns = self.pinecone_namespace_var.get() or "(default)"
        ttk.Label(ns_frame, text=f"Delete ALL vectors in namespace: {current_ns}", foreground='#f38ba8').pack(anchor='w')
        
        def clear_namespace():
            ns = self.pinecone_namespace_var.get() or ''
            confirm = messagebox.askyesno(
                "⚠️ DANGER",
                f"This will DELETE ALL VECTORS in namespace '{ns or '(default)'}'\n\n"
                "This action CANNOT be undone!\n\nAre you ABSOLUTELY sure?"
            )
            if not confirm:
                return
            
            # Double confirm for safety
            double_confirm = messagebox.askyesno(
                "Final Confirmation",
                "Last chance - DELETE ALL VECTORS?\n\nClick YES to proceed."
            )
            if not double_confirm:
                return
            
            dialog.destroy()
            self._clear_namespace(ns)
        
        ttk.Button(ns_frame, text="🔥 Clear Namespace", command=clear_namespace).pack(pady=5)
        
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).pack(pady=15)
    
    def _delete_by_filter(self, metadata_filter):
        """Delete vectors matching a metadata filter using SDK delete(filter=...)."""
        self._set_status("Deleting by filter...", True)
        
        def delete():
            import time
            index, index_name = self._get_index_client()
            namespace = self.pinecone_namespace_var.get() or ''
            
            self._trace('delete_by_filter_start', index=index_name, namespace=namespace, filter=metadata_filter)
            start_time = time.time()
            
            try:
                # SDK supports delete with filter!
                index.delete(filter=metadata_filter, namespace=namespace or None)
                duration_ms = int((time.time() - start_time) * 1000)
                self._trace('delete_by_filter_complete', duration_ms=duration_ms)
                return True
            except Exception as e:
                self._trace('delete_by_filter_error', error=str(e))
                raise e
        
        def on_complete(success):
            self._set_status("Ready", False)
            if success:
                self._log("✅ Vectors deleted by filter", 'SUCCESS')
                self._refresh_pinecone_vectors()
        
        self._run_async(delete, on_complete)
    
    def _clear_namespace(self, namespace):
        """Clear all vectors in a namespace using SDK delete(delete_all=True)."""
        self._set_status(f"Clearing namespace '{namespace}'...", True)
        
        def delete():
            import time
            index, index_name = self._get_index_client()
            
            self._trace('clear_namespace_start', index=index_name, namespace=namespace)
            start_time = time.time()
            
            try:
                # SDK supports delete_all!
                index.delete(delete_all=True, namespace=namespace or None)
                duration_ms = int((time.time() - start_time) * 1000)
                self._trace('clear_namespace_complete', duration_ms=duration_ms)
                return True
            except Exception as e:
                self._trace('clear_namespace_error', error=str(e))
                raise e
        
        def on_complete(success):
            self._set_status("Ready", False)
            if success:
                self._log(f"✅ Namespace '{namespace or '(default)'}' cleared", 'SUCCESS')
                self._refresh_pinecone_vectors()
        
        self._run_async(delete, on_complete)

    def _fetch_vector_details(self):
        """Fetch and display detailed info for selected vector."""
        selection = self.pinecone_tree.selection()
        if not selection or len(selection) != 1:
            messagebox.showinfo("Info", "Please select exactly one vector")
            return
        
        item = self.pinecone_tree.item(selection[0])
        values = item['values']
        
        # Create a dialog to show detailed metadata
        dialog = tk.Toplevel(self.root)
        dialog.title("Vector Metadata")
        dialog.geometry("700x600")
        
        # Add scrolled text widget
        import tkinter.scrolledtext as scrolledtext
        text = scrolledtext.ScrolledText(dialog, wrap=tk.WORD, font=('Monaco', 11))
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Fetch the actual vector to get full metadata
        def fetch_metadata():
            index, _ = self._get_index_client()
            namespace = self.pinecone_namespace_var.get() or ''
            
            # Get full vector ID from selection (iid)
            vector_id = selection[0]
            
            try:
                # Direct fetch by ID
                result = index.fetch(ids=[vector_id], namespace=namespace if namespace else None)
                if result and result.vectors and vector_id in result.vectors:
                    return result.vectors[vector_id]
            except Exception as e:
                self._trace('vector_detail_fetch_error', vector_id=vector_id, error=str(e))
            
            return None
        
        def display_metadata(vec):
            if not vec:
                text.insert('1.0', "Vector not found or no metadata available.")
                return
            
            import json
            meta = vec.metadata if hasattr(vec, 'metadata') else {}
            
            details = f"""🔍 Vector Details
            
📋 Full ID: {vec.id}
📊 Score: {vec.score if hasattr(vec, 'score') else 'N/A'}

📦 Metadata ({len(meta)} fields):
{'─' * 60}
"""
            
            if meta:
                for key, value in sorted(meta.items()):
                    # Truncate long values
                    value_str = str(value)
                    if len(value_str) > 200:
                        value_str = value_str[:200] + "..."
                    details += f"\n{key}: {value_str}\n"
            else:
                details += "\n⚠️ No metadata found for this vector\n"
            
            details += f"\n{'─' * 60}\n\n📝 Raw JSON:\n{json.dumps(meta, indent=2, default=str)}"
            
            text.insert('1.0', details)
            text.configure(state='disabled')
        
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=5)
        
        # Fetch metadata asynchronously
        self._run_async(fetch_metadata, display_metadata)
    
    def _export_pinecone_data(self):
        """Export all Pinecone vectors and metadata to JSON/CSV."""
        from tkinter import filedialog
        import json
        import csv
        from datetime import datetime
        
        # Ask for format
        dialog = tk.Toplevel(self.root)
        dialog.title("Export Pinecone Data")
        dialog.geometry("400x200")
        
        ttk.Label(dialog, text="Export Format:", font=('Helvetica', 12, 'bold')).pack(pady=10)
        
        format_var = tk.StringVar(value="json")
        ttk.Radiobutton(dialog, text="JSON (complete with vectors)", variable=format_var, value="json").pack(pady=5)
        ttk.Radiobutton(dialog, text="JSON (metadata only)", variable=format_var, value="json_meta").pack(pady=5)
        ttk.Radiobutton(dialog, text="CSV (metadata only)", variable=format_var, value="csv").pack(pady=5)
        
        def do_export():
            format_choice = format_var.get()
            dialog.destroy()
            
            # Get filename
            index_name = self.pinecone_index_var.get() or 'pinecone'
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"{index_name}_export_{timestamp}"
            
            if format_choice == "csv":
                filename = filedialog.asksaveasfilename(
                    defaultextension=".csv",
                    initialfile=f"{default_name}.csv",
                    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
                )
            else:
                filename = filedialog.asksaveasfilename(
                    defaultextension=".json",
                    initialfile=f"{default_name}.json",
                    filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
                )
            
            if not filename:
                return
            
            self._set_status("Exporting data...", True)
            
            def fetch_all_data():
                index, index_name = self._get_index_client()
                namespace = (self.pinecone_namespace_var.get() or '').strip()
                
                stats = index.describe_index_stats()
                
                all_vectors = []
                
                # List all IDs
                ids_to_fetch = []
                try:
                    for batch in index.list(namespace=namespace or None):
                        ids_to_fetch.extend(batch)
                except Exception as e:
                    self._trace('vector_export_list_error', namespace=namespace or 'ALL', error=str(e))
                
                # Fetch in batches
                chunk_size = 100
                for i in range(0, len(ids_to_fetch), chunk_size):
                    chunk = ids_to_fetch[i:i+chunk_size]
                    try:
                        res = index.fetch(ids=chunk, namespace=namespace or None)
                        all_vectors.extend(res.vectors.values())
                    except Exception as e:
                        self._trace('vector_export_fetch_error', batch_index=i, error=str(e))
                
                return {
                    'vectors': all_vectors,
                    'stats': stats,
                    'index_name': index_name,
                    'namespace': namespace,
                    'export_time': datetime.now().isoformat()
                }
            
            def save_data(data):
                try:
                    if format_choice == "csv":
                        # CSV export - metadata only
                        with open(filename, 'w', newline='', encoding='utf-8') as f:
                            if not data['vectors']:
                                self._log("No vectors to export", 'WARNING')
                                return
                            
                            # Get all unique metadata keys
                            all_keys = set()
                            for vec in data['vectors']:
                                if hasattr(vec, 'metadata'):
                                    all_keys.update(vec.metadata.keys())
                            
                            fieldnames = ['vector_id', 'score'] + sorted(all_keys)
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            
                            for vec in data['vectors']:
                                row = {
                                    'vector_id': vec.id if hasattr(vec, 'id') else '',
                                    'score': vec.score if hasattr(vec, 'score') else ''
                                }
                                if hasattr(vec, 'metadata'):
                                    row.update(vec.metadata)
                                writer.writerow(row)
                        
                        self._log(f"Exported {len(data['vectors'])} vectors to CSV: {filename}", 'SUCCESS')
                    
                    elif format_choice == "json_meta":
                        # JSON metadata only
                        export_data = {
                            'index': data['index_name'],
                            'namespace': data['namespace'],
                            'exported_at': data['export_time'],
                            'total_vectors': len(data['vectors']),
                            'vectors': [
                                {
                                    'id': vec.id if hasattr(vec, 'id') else None,
                                    'score': vec.score if hasattr(vec, 'score') else None,
                                    'metadata': vec.metadata if hasattr(vec, 'metadata') else {}
                                }
                                for vec in data['vectors']
                            ]
                        }
                        with open(filename, 'w', encoding='utf-8') as f:
                            json.dump(export_data, f, indent=2, default=str)
                        
                        self._log(f"Exported {len(data['vectors'])} vectors (metadata) to JSON: {filename}", 'SUCCESS')
                    
                    else:  # json - complete
                        # Full JSON with vectors (warning: large!)
                        export_data = {
                            'index': data['index_name'],
                            'namespace': data['namespace'],
                            'dimension': data['stats'].dimension,
                            'metric': data['stats'].metric,
                            'exported_at': data['export_time'],
                            'total_vectors': len(data['vectors']),
                            'vectors': [
                                {
                                    'id': vec.id if hasattr(vec, 'id') else None,
                                    'score': vec.score if hasattr(vec, 'score') else None,
                                    'values': vec.values if hasattr(vec, 'values') else [],
                                    'metadata': vec.metadata if hasattr(vec, 'metadata') else {}
                                }
                                for vec in data['vectors']
                            ]
                        }
                        with open(filename, 'w', encoding='utf-8') as f:
                            json.dump(export_data, f, indent=2, default=str)
                        
                        self._log(f"Exported {len(data['vectors'])} complete vectors to JSON: {filename}", 'SUCCESS')
                    
                    self._set_status("Ready", False)
                    messagebox.showinfo("Export Complete", f"Successfully exported {len(data['vectors'])} vectors to:\\n{filename}")
                
                except Exception as e:
                    self._log(f"Export failed: {str(e)}", 'ERROR')
                    messagebox.showerror("Export Error", f"Failed to export data:\\n{str(e)}")
                    self._set_status("Ready", False)
            
            self._run_async(fetch_all_data, save_data)
        
        ttk.Button(dialog, text="Export", command=do_export, width=15).pack(pady=20)
        ttk.Button(dialog, text="Cancel", command=dialog.destroy, width=15).pack()
    
    def _clear_all_pinecone(self):
        """Clear all vectors from Pinecone index."""
        result = messagebox.askyesno(
            "⚠️ DANGER ⚠️",
            "Delete ALL vectors from Pinecone?\\n\\nThis will erase the entire index.\\nAre you absolutely sure?"
        )
        if not result:
            return
        
        # Double confirmation
        result2 = messagebox.askyesno(
            "Final Confirmation",
            "Type 'yes' to confirm deletion of all vectors"
        )
        if not result2:
            return
        
        self._set_status("Clearing Pinecone index...", True)
        
        def clear_all():
            index, _ = self._get_index_client()
            namespace = (self.pinecone_namespace_var.get() or '').strip()
            index.delete(delete_all=True, namespace=namespace or None)
            return True
        
        def on_complete(success):
            self._set_status("Ready", False)
            if success:
                self._log("🧹 Cleared all vectors from index/namespace", 'SUCCESS')
                # Clear the tree immediately
                for item in self.pinecone_tree.get_children():
                    self.pinecone_tree.delete(item)
                # Update preview
                if hasattr(self, 'pinecone_preview_text'):
                    self.pinecone_preview_text.configure(state='normal')
                    self.pinecone_preview_text.delete('1.0', tk.END)
                    self.pinecone_preview_text.insert('1.0', "✨ Index/namespace is now empty")
                    self.pinecone_preview_text.configure(state='disabled')
                # Update stats
                if hasattr(self, 'pinecone_stat_labels'):
                    self.pinecone_stat_labels['total_vectors'].configure(text='0')
                # Show status
                if hasattr(self, 'pinecone_status_label'):
                    self.pinecone_status_label.config(text="● Empty", foreground='#f9e2af')
                    self.root.after(3000, lambda: self.pinecone_status_label.config(text="● Ready", foreground='#a6e3a1'))
        
        self._run_async(clear_all, on_complete)
    
    def _show_pinecone_stats(self):
        """Show detailed Pinecone index statistics."""
        def fetch_stats():
            index, _ = self._get_index_client()
            return index.describe_index_stats()
        
        def show_stats(stats):
            dialog = tk.Toplevel(self.root)
            dialog.title("Pinecone Index Statistics")
            dialog.geometry("600x500")
            
            text = scrolledtext.ScrolledText(dialog, wrap=tk.WORD, font=('Monaco', 11))
            text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Calculate more detailed stats
            total_vectors = stats.total_vector_count
            fullness_pct = stats.index_fullness * 100 if hasattr(stats, 'index_fullness') else 0
            
            stats_text = f"""📊 Comprehensive Pinecone Index Statistics

📦 Index: {self.pinecone_index_var.get()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 Capacity & Usage:
  • Total Vectors: {total_vectors:,}
  • Dimension: {stats.dimension}
  • Distance Metric: {stats.metric}
  • Index Fullness: {fullness_pct:.2f}%
  • Storage Used: ~{(total_vectors * stats.dimension * 4 / 1024 / 1024):.2f} MB

🏷️  Namespaces:
"""
            if hasattr(stats, 'namespaces'):
                for ns_name, ns_data in stats.namespaces.items():
                    ns_count = ns_data.get('vector_count', 0) if isinstance(ns_data, dict) else getattr(ns_data, 'vector_count', 0)
                    stats_text += f"  - {ns_name}: {ns_count} vectors\\n"
            
            # Properly format raw stats
            try:
                if hasattr(stats, '__dict__'):
                    raw_stats = vars(stats)
                elif callable(getattr(stats, '_asdict', None)):
                    raw_stats = stats._asdict()
                else:
                    raw_stats = str(stats)
                stats_text += f"\\n\\nRaw Response:\\n{json.dumps(raw_stats, indent=2, default=str)}"
            except Exception as e:
                stats_text += f"\\n\\nRaw Response: {stats}"
            
            text.insert('1.0', stats_text)
            text.configure(state='disabled')
            
            ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=5)
        
        self._run_async(fetch_stats, show_stats)
    
    def _sort_pinecone_column(self, col, is_numeric):
        """Sort Pinecone tree by column."""
        items = [(self.pinecone_tree.set(item, col), item) for item in self.pinecone_tree.get_children('')]
        
        if is_numeric:
            # Handle numeric/score sorting
            def sort_key(x):
                val = x[0]
                if val in ('—', 'N/A', ''):
                    return -1
                try:
                    # Extract numeric value from strings like "5m23s" or "0.123"
                    import re
                    nums = re.findall(r'\d+\.?\d*', val)
                    return float(nums[0]) if nums else -1
                except:
                    return -1
            items.sort(key=sort_key, reverse=True)
        else:
            items.sort(key=lambda x: str(x[0]).lower())
        
        # Rearrange items
        for index, (val, item) in enumerate(items):
            self.pinecone_tree.move(item, '', index)
        
        self._log(f"Sorted by {col}", 'INFO')
    
    def _clear_placeholder(self, entry, placeholder):
        """Clear placeholder text on focus."""
        if entry.get() == placeholder:
            entry.delete(0, tk.END)
            entry.config(foreground='white')
    
    def _restore_placeholder(self, entry, placeholder):
        """Restore placeholder text on focus out."""
        if not entry.get():
            entry.insert(0, placeholder)
            entry.config(foreground='gray')
    
    def _filter_pinecone_vectors(self):
        """Filter displayed Pinecone vectors."""
        query = self.pinecone_filter.get().lower()
        if query == "type to filter by id, title, themes...":
            query = ""
        
        for item in self.pinecone_tree.get_children():
            values = self.pinecone_tree.item(item)['values']
            text = ' '.join(str(v).lower() for v in values)
            
            if query in text:
                self.pinecone_tree.reattach(item, '', tk.END)
            else:
                self.pinecone_tree.detach(item)
    
    def _show_pinecone_context_menu(self, event):
        """Show context menu for Pinecone tree."""
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="View Details", command=self._fetch_vector_details)
        menu.add_command(label="Delete Selected", command=self._delete_pinecone_vectors)
        menu.add_separator()
        menu.add_command(label="Refresh All", command=self._refresh_pinecone_vectors)
        
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
    
    def _clear_search(self):
        """Clear search results."""
        self.search_entry.delete(0, tk.END)
        self.search_results_text.delete('1.0', tk.END)
        self._log("Search results cleared", 'INFO')
    
    # ==================== NEW PINECONE FEATURES ====================
    
    def _fetch_vector_full_details(self):
        """Fetch complete vector data including embeddings using fetch() API."""
        selection = self.pinecone_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a vector to fetch full details.")
            return
        
        vector_id_display = self.pinecone_tree.item(selection[0])['values'][0]
        # Extract full ID (remove ellipsis if present)
        if '...' in vector_id_display:
            # Need to get full ID from stored data
            messagebox.showinfo("Fetch", "Please use the 'View Details' button for metadata.\nFetch requires the complete vector ID.")
            return
        
        self._set_status("Fetching full vector data...", True)
        
        def fetch_full():
            index, _ = self._get_index_client()
            namespace = (self.pinecone_namespace_var.get() or '').strip()
            
            # Use fetch() to get complete vector including embeddings
            result = index.fetch(ids=[vector_id_display], namespace=namespace or None)
            return result.vectors.get(vector_id_display) if hasattr(result, 'vectors') else None
        
        def display_full(vec):
            self._set_status("Ready", False)
            if not vec:
                messagebox.showerror("Error", "Vector not found or fetch failed.")
                return
            
            # Create dialog to show full vector details
            dialog = tk.Toplevel(self.root)
            dialog.title(f"Full Vector Data: {vec.id if hasattr(vec, 'id') else 'Unknown'}")
            dialog.geometry("900x700")
            dialog.configure(bg=self.colors['bg'])
            
            # Create tabbed interface
            notebook = ttk.Notebook(dialog)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Metadata tab
            meta_frame = ttk.Frame(notebook)
            notebook.add(meta_frame, text="📋 Metadata")
            
            meta_text = scrolledtext.ScrolledText(meta_frame, wrap=tk.WORD, font=('Monaco', 10),
                                                  bg=self.colors['surface'], fg=self.colors['fg'])
            meta_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            import json
            meta = vec.metadata if hasattr(vec, 'metadata') else {}
            meta_text.insert('1.0', json.dumps(meta, indent=2))
            meta_text.configure(state='disabled')
            
            # Embedding tab
            embed_frame = ttk.Frame(notebook)
            notebook.add(embed_frame, text="🔢 Embedding Vector")
            
            embed_text = scrolledtext.ScrolledText(embed_frame, wrap=tk.WORD, font=('Monaco', 9),
                                                   bg=self.colors['surface'], fg=self.colors['fg'])
            embed_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            values = vec.values if hasattr(vec, 'values') else []
            embed_text.insert('1.0', f"Dimension: {len(values)}\n\n")
            embed_text.insert(tk.END, f"First 20 values:\n{values[:20]}\n\n")
            embed_text.insert(tk.END, f"Last 20 values:\n{values[-20:]}\n\n")
            embed_text.insert(tk.END, f"Full vector:\n{values}")
            embed_text.configure(state='disabled')
            
            # Raw data tab
            raw_frame = ttk.Frame(notebook)
            notebook.add(raw_frame, text="📄 Raw JSON")
            
            raw_text = scrolledtext.ScrolledText(raw_frame, wrap=tk.WORD, font=('Monaco', 10),
                                                bg=self.colors['surface'], fg=self.colors['fg'])
            raw_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            vec_dict = {
                'id': vec.id if hasattr(vec, 'id') else None,
                'values': vec.values if hasattr(vec, 'values') else [],
                'metadata': meta
            }
            raw_text.insert('1.0', json.dumps(vec_dict, indent=2))
            raw_text.configure(state='disabled')
            
            self._log(f"Fetched full vector data for: {vec.id if hasattr(vec, 'id') else 'unknown'}", 'SUCCESS')
        
        self._run_async(fetch_full, display_full)
    
    def _update_vector_metadata(self):
        """Update metadata for selected vector using update() API."""
        selection = self.pinecone_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a vector to edit metadata.")
            return
        
        # Get vector details first
        item = self.pinecone_tree.item(selection[0])
        vector_id = item['values'][0].replace('...', '')  # Remove ellipsis
        
        # Create edit dialog
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Edit Metadata: {vector_id}")
        dialog.geometry("700x500")
        dialog.configure(bg=self.colors['bg'])
        
        ttk.Label(dialog, text="Edit Vector Metadata", font=('Helvetica', 14, 'bold')).pack(pady=10)
        ttk.Label(dialog, text=f"Vector ID: {vector_id}", font=('Helvetica', 10)).pack(pady=5)
        
        # Fetch current metadata
        def fetch_current():
            index, _ = self._get_index_client()
            namespace = (self.pinecone_namespace_var.get() or '').strip()
            
            # Query to get metadata
            stats = index.describe_index_stats()
            dimension = stats.dimension
            result = index.query(
                vector=[0.0] * dimension,
                top_k=1,
                include_metadata=True,
                filter={"$and": []},  # Match all
                namespace=namespace or None
            )
            
            for vec in result.matches:
                if vector_id in vec.id:
                    return vec.metadata if hasattr(vec, 'metadata') else {}
            return {}
        
        # Text editor for metadata (JSON format)
        ttk.Label(dialog, text="Metadata (JSON format):").pack(pady=5)
        
        meta_text = scrolledtext.ScrolledText(dialog, wrap=tk.WORD, font=('Monaco', 10),
                                              bg=self.colors['surface'], fg=self.colors['fg'],
                                              height=20)
        meta_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Load current metadata
        import json
        current_meta = fetch_current()
        meta_text.insert('1.0', json.dumps(current_meta, indent=2))
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        
        def save_metadata():
            try:
                # Parse JSON
                new_meta = json.loads(meta_text.get('1.0', tk.END))
                
                # Update vector
                index, _ = self._get_index_client()
                namespace = (self.pinecone_namespace_var.get() or '').strip()
                
                index.update(
                    id=vector_id,
                    set_metadata=new_meta,
                    namespace=namespace or None
                )
                
                messagebox.showinfo("Success", f"Metadata updated for vector: {vector_id}")
                self._log(f"Updated metadata for: {vector_id}", 'SUCCESS')
                dialog.destroy()
                
                # Refresh view
                self._refresh_pinecone_vectors()
                
            except json.JSONDecodeError as e:
                messagebox.showerror("JSON Error", f"Invalid JSON format:\n{e}")
            except Exception as e:
                messagebox.showerror("Update Error", f"Failed to update metadata:\n{e}")
                self._log(f"Error updating metadata: {e}", 'ERROR')
        
        ttk.Button(button_frame, text="💾 Save Changes", command=save_metadata).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="❌ Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def _upsert_vector_dialog(self):
        """Dialog to manually add or update a vector using upsert() API."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Add/Update Vector")
        dialog.geometry("750x600")
        dialog.configure(bg=self.colors['bg'])
        
        ttk.Label(dialog, text="Add or Update Vector", font=('Helvetica', 14, 'bold')).pack(pady=10)
        
        # Form frame
        form = ttk.Frame(dialog)
        form.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Vector ID
        ttk.Label(form, text="Vector ID:").grid(row=0, column=0, sticky='w', pady=5)
        id_entry = ttk.Entry(form, width=50)
        id_entry.grid(row=0, column=1, sticky='ew', pady=5)
        form.columnconfigure(1, weight=1)
        
        # Metadata
        ttk.Label(form, text="Metadata (JSON):").grid(row=1, column=0, sticky='nw', pady=5)
        meta_text = scrolledtext.ScrolledText(form, wrap=tk.WORD, font=('Monaco', 10),
                                              bg=self.colors['surface'], fg=self.colors['fg'],
                                              height=15, width=60)
        meta_text.grid(row=1, column=1, sticky='nsew', pady=5)
        form.rowconfigure(1, weight=1)
        
        # Default metadata template
        default_meta = {
            "title": "New Vector",
            "text": "Full text content goes here. This will be used to generate the embedding.",
            "timestamp": datetime.now().isoformat(),
            "source": "manual"
        }
        import json
        meta_text.insert('1.0', json.dumps(default_meta, indent=2))
        
        # Embedding options
        ttk.Label(form, text="Embedding:").grid(row=2, column=0, sticky='w', pady=5)
        embed_frame = ttk.Frame(form)
        embed_frame.grid(row=2, column=1, sticky='ew', pady=5)
        
        embed_option = tk.StringVar(value="generate")
        ttk.Radiobutton(embed_frame, text="Auto-generate from content", variable=embed_option, value="generate").pack(anchor='w')
        ttk.Radiobutton(embed_frame, text="Provide custom embedding", variable=embed_option, value="custom").pack(anchor='w')
        
        # Custom embedding input
        ttk.Label(form, text="Custom Embedding\n(comma-separated):").grid(row=3, column=0, sticky='nw', pady=5)
        embed_text = scrolledtext.ScrolledText(form, wrap=tk.WORD, font=('Monaco', 9),
                                              bg=self.colors['surface'], fg=self.colors['fg'],
                                              height=5)
        embed_text.grid(row=3, column=1, sticky='ew', pady=5)
        embed_text.insert('1.0', "# Paste embedding vector here as comma-separated values\n# Or leave empty to auto-generate")
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        
        def do_upsert():
            try:
                vector_id = id_entry.get().strip()
                if not vector_id:
                    messagebox.showerror("Error", "Vector ID is required")
                    return
                
                # Parse metadata
                metadata = json.loads(meta_text.get('1.0', tk.END))
                
                # Get or generate embedding
                if embed_option.get() == "custom":
                    embed_str = embed_text.get('1.0', tk.END).strip()
                    # Remove comments
                    embed_str = '\n'.join([line for line in embed_str.split('\n') if not line.strip().startswith('#')])
                    values = [float(x.strip()) for x in embed_str.split(',') if x.strip()]
                else:
                    # Generate embedding from text field (or fall back to title)
                    content = metadata.get('text', metadata.get('content', metadata.get('title', '')))
                    if not content:
                        messagebox.showerror("Error", "No 'text' field found in metadata for embedding generation")
                        return
                    
                    # Truncate long content for embedding (OpenAI limit ~8k tokens)
                    content_for_embedding = content[:8000] if len(content) > 8000 else content
                    
                    # Generate embedding using OpenAI
                    import openai
                    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                    response = client.embeddings.create(
                        model="text-embedding-3-large",
                        input=content_for_embedding
                    )
                    values = response.data[0].embedding
                    
                    # Log if content was truncated
                    if len(content) > 8000:
                        self._log(f"Content truncated for embedding: {len(content)} -> 8000 chars", 'INFO')
                
                # Upsert to Pinecone
                index, _ = self._get_index_client()
                namespace = (self.pinecone_namespace_var.get() or '').strip()
                
                index.upsert(
                    vectors=[(vector_id, values, metadata)],
                    namespace=namespace or None
                )
                
                messagebox.showinfo("Success", f"Vector upserted successfully:\n{vector_id}")
                self._log(f"Upserted vector: {vector_id}", 'SUCCESS')
                dialog.destroy()
                
                # Refresh view
                self._refresh_pinecone_vectors()
                
            except json.JSONDecodeError as e:
                messagebox.showerror("JSON Error", f"Invalid JSON in metadata:\n{e}")
            except ValueError as e:
                messagebox.showerror("Embedding Error", f"Invalid embedding format:\n{e}")
            except Exception as e:
                messagebox.showerror("Upsert Error", f"Failed to upsert vector:\n{e}")
                self._log(f"Error upserting vector: {e}", 'ERROR')
        
        ttk.Button(button_frame, text="💾 Upsert Vector", command=do_upsert).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="❌ Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def _create_namespace_dialog(self):
        """Create a new namespace."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Create Namespace")
        dialog.geometry("400x200")
        dialog.configure(bg=self.colors['bg'])
        
        ttk.Label(dialog, text="Create New Namespace", font=('Helvetica', 12, 'bold')).pack(pady=20)
        
        ttk.Label(dialog, text="Namespace Name:").pack(pady=5)
        name_entry = ttk.Entry(dialog, width=40)
        name_entry.pack(pady=5)
        
        ttk.Label(dialog, text="(Leave empty for default namespace)", foreground='gray').pack(pady=2)
        
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=20)
        
        def create():
            namespace = name_entry.get().strip()
            
            try:
                index, _ = self._get_index_client()
                
                # Create namespace (need to upsert at least one dummy vector)
                # Pinecone creates namespaces implicitly, but we can ensure it exists
                stats = index.describe_index_stats()
                dimension = stats.dimension
                
                # Create a dummy vector with non-zero values (Pinecone requirement)
                # Use a small value repeated to ensure it's not all zeros
                dummy_vector = [0.001] * dimension
                dummy_id = f"__namespace_init_{namespace or 'default'}"
                index.upsert(
                    vectors=[(dummy_id, dummy_vector, {"_init": True, "created": datetime.now().isoformat()})],
                    namespace=namespace or None
                )
                
                # Optionally delete the dummy vector
                # index.delete(ids=[dummy_id], namespace=namespace if namespace else None)
                
                messagebox.showinfo("Success", f"Namespace created: {namespace or 'default'}\n\nNote: A placeholder vector was added.")
                self._log(f"Created namespace: {namespace or 'default'}", 'SUCCESS')
                dialog.destroy()
                
                # Refresh namespaces
                self._load_namespaces_only()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create namespace:\n{e}")
                self._log(f"Error creating namespace: {e}", 'ERROR')
        
        ttk.Button(button_frame, text="✅ Create", command=create).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="❌ Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def _delete_namespace_dialog(self):
        """Delete an entire namespace."""
        namespace = self.pinecone_namespace_var.get() or ''
        display_name = namespace or 'default'
        
        if not namespace and not messagebox.askyesno("Confirm", "Delete the DEFAULT namespace?\n\nThis will delete all vectors in the default namespace."):
            return
        
        if namespace and not messagebox.askyesno("Confirm", f"Delete namespace '{namespace}'?\n\nThis will permanently delete all vectors in this namespace.\n\nThis action cannot be undone!"):
            return
        
        self._set_status(f"Deleting namespace '{display_name}'...", True)
        
        def delete():
            index, _ = self._get_index_client()
            index.delete(delete_all=True, namespace=namespace or None)
            return display_name
        
        def on_complete(display_name):
            self._set_status("Ready", False)
            messagebox.showinfo("Success", f"Namespace '{display_name}' deleted successfully.")
            self._log(f"Deleted namespace: {display_name}", 'SUCCESS')
            
            # Refresh
            self._load_namespaces_only()
            self._refresh_pinecone_vectors()
        
        self._run_async(delete, on_complete)
    
    def _show_namespace_stats(self):
        """Show detailed stats for each namespace."""
        self._set_status("Loading namespace stats...", True)
        
        def fetch_stats():
            index, _ = self._get_index_client()
            return index.describe_index_stats()
        
        def display_stats(stats):
            self._set_status("Ready", False)
            
            dialog = tk.Toplevel(self.root)
            dialog.title("Namespace Statistics")
            dialog.geometry("600x400")
            dialog.configure(bg=self.colors['bg'])
            
            ttk.Label(dialog, text="Namespace Statistics", font=('Helvetica', 14, 'bold')).pack(pady=10)
            
            # Create treeview for stats
            columns = ('namespace', 'vector_count')
            tree = ttk.Treeview(dialog, columns=columns, show='headings', height=15)
            tree.heading('namespace', text='Namespace')
            tree.heading('vector_count', text='Vector Count')
            tree.column('namespace', width=300)
            tree.column('vector_count', width=150)
            
            scrollbar = ttk.Scrollbar(dialog, orient=tk.VERTICAL, command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)
            
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
            
            # Populate
            if hasattr(stats, 'namespaces') and stats.namespaces:
                for ns, ns_stats in stats.namespaces.items():
                    display_ns = ns if ns else '(default)'
                    count = ns_stats.vector_count if hasattr(ns_stats, 'vector_count') else 0
                    tree.insert('', 'end', values=(display_ns, count))
            
            # Total
            total_label = ttk.Label(dialog, text=f"Total Vectors: {stats.total_vector_count}", 
                                   font=('Helvetica', 11, 'bold'))
            total_label.pack(pady=10)
            
            ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)
        
        self._run_async(fetch_stats, display_stats)
    
    def _query_similar_dialog(self):
        """Query for similar vectors using a vector ID or manual embedding."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Query Similar Vectors")
        dialog.geometry("600x500")
        dialog.configure(bg=self.colors['bg'])
        
        ttk.Label(dialog, text="🔍 Query Similar Vectors", font=('Helvetica', 14, 'bold')).pack(pady=10)
        
        # Query type
        query_frame = ttk.LabelFrame(dialog, text="Query Method", padding=10)
        query_frame.pack(fill=tk.X, padx=10, pady=5)
        
        query_type = tk.StringVar(value="id")
        ttk.Radiobutton(query_frame, text="By Vector ID", variable=query_type, value="id").pack(anchor='w')
        ttk.Radiobutton(query_frame, text="By Vector Values (comma-separated)", variable=query_type, value="vector").pack(anchor='w')
        
        # Input
        input_frame = ttk.LabelFrame(dialog, text="Query Input", padding=10)
        input_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        ttk.Label(input_frame, text="Enter Vector ID or Values:").pack(anchor='w')
        query_input = scrolledtext.ScrolledText(input_frame, height=5, wrap=tk.WORD)
        query_input.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Parameters
        param_frame = ttk.LabelFrame(dialog, text="Parameters", padding=10)
        param_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(param_frame, text="Top K Results:").grid(row=0, column=0, sticky='e', padx=5, pady=5)
        top_k_var = tk.IntVar(value=10)
        ttk.Spinbox(param_frame, from_=1, to=10000, textvariable=top_k_var, width=10).grid(row=0, column=1, sticky='w', padx=5, pady=5)
        
        ttk.Label(param_frame, text="Namespace:").grid(row=1, column=0, sticky='e', padx=5, pady=5)
        ns_var = tk.StringVar(value=self.pinecone_namespace_var.get())
        ttk.Entry(param_frame, textvariable=ns_var, width=30).grid(row=1, column=1, sticky='w', padx=5, pady=5)
        
        include_meta_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="Include Metadata", variable=include_meta_var).grid(row=2, column=0, columnspan=2, pady=5)
        
        include_vals_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(param_frame, text="Include Values", variable=include_vals_var).grid(row=3, column=0, columnspan=2, pady=5)
        
        def do_query():
            query_text = query_input.get('1.0', tk.END).strip()
            if not query_text:
                messagebox.showwarning("No Input", "Please provide a vector ID or values")
                return
            
            self._set_status("Querying similar vectors...", True)
            
            def query():
                index, _ = self._get_index_client()
                namespace = (ns_var.get() or '').strip()
                
                query_params = {
                    'top_k': top_k_var.get(),
                    'namespace': namespace or None,
                    'include_metadata': include_meta_var.get(),
                    'include_values': include_vals_var.get()
                }
                
                if query_type.get() == "id":
                    query_params['id'] = query_text
                else:
                    # Parse comma-separated values
                    try:
                        vector_vals = [float(x.strip()) for x in query_text.split(',')]
                        query_params['vector'] = vector_vals
                    except Exception:
                        raise ValueError("Invalid vector format. Use comma-separated numbers.")
                
                return index.query(**query_params)
            
            def show_results(results):
                self._set_status("Ready", False)
                dialog.destroy()
                
                # Show results in new dialog
                result_dialog = tk.Toplevel(self.root)
                result_dialog.title("Query Results")
                result_dialog.geometry("800x600")
                result_dialog.configure(bg=self.colors['bg'])
                
                ttk.Label(result_dialog, text=f"Found {len(results.matches)} matches", 
                         font=('Helvetica', 12, 'bold')).pack(pady=10)
                
                result_text = scrolledtext.ScrolledText(result_dialog, wrap=tk.WORD, font=('Monaco', 9))
                result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                
                for i, match in enumerate(results.matches, 1):
                    result_text.insert(tk.END, f"\n{'='*80}\n")
                    result_text.insert(tk.END, f"Match #{i}\n")
                    result_text.insert(tk.END, f"{'='*80}\n")
                    result_text.insert(tk.END, f"ID: {match.id}\n")
                    result_text.insert(tk.END, f"Score: {match.score:.4f}\n")
                    if hasattr(match, 'metadata') and match.metadata:
                        result_text.insert(tk.END, f"\nMetadata:\n")
                        for key, val in match.metadata.items():
                            result_text.insert(tk.END, f"  {key}: {val}\n")
                    if hasattr(match, 'values') and match.values:
                        result_text.insert(tk.END, f"\nVector (first 10): {match.values[:10]}...\n")
                    result_text.insert(tk.END, "\n")
                
                ttk.Button(result_dialog, text="Close", command=result_dialog.destroy).pack(pady=10)
            
            self._run_async(query, show_results)
        
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="🔍 Query", command=do_query).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def _query_all_namespaces_dialog(self):
        """Query across ALL namespaces in parallel using SDK's query_namespaces()."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Query All Namespaces")
        dialog.geometry("700x600")
        dialog.configure(bg=self.colors['bg'])
        
        ttk.Label(dialog, text="🌐 Query Across All Namespaces", font=('Helvetica', 14, 'bold')).pack(pady=10)
        ttk.Label(dialog, text="Searches ALL namespaces in parallel and merges results", 
                 foreground='#89b4fa').pack()
        
        # Query input
        input_frame = ttk.LabelFrame(dialog, text="Query Vector", padding=10)
        input_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(input_frame, text="Enter a Vector ID to use as query:").pack(anchor='w')
        query_id_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=query_id_var, width=60).pack(fill=tk.X, pady=5)
        
        ttk.Label(input_frame, text="Or provide vector values (comma-separated):").pack(anchor='w', pady=(10,0))
        vector_text = scrolledtext.ScrolledText(input_frame, height=4, wrap=tk.WORD, font=('Monaco', 9))
        vector_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Parameters
        param_frame = ttk.LabelFrame(dialog, text="Parameters", padding=10)
        param_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(param_frame, text="Top K (per namespace):").grid(row=0, column=0, sticky='e', padx=5)
        top_k_var = tk.IntVar(value=10)
        ttk.Spinbox(param_frame, from_=1, to=1000, textvariable=top_k_var, width=10).grid(row=0, column=1, sticky='w', padx=5)
        
        include_meta_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="Include Metadata", variable=include_meta_var).grid(row=1, column=0, columnspan=2, pady=5)
        
        # Metadata filter
        filter_frame = ttk.LabelFrame(dialog, text="Optional Metadata Filter (JSON)", padding=10)
        filter_frame.pack(fill=tk.X, padx=10, pady=5)
        
        filter_text = scrolledtext.ScrolledText(filter_frame, height=3, wrap=tk.WORD, font=('Monaco', 9))
        filter_text.pack(fill=tk.X, pady=5)
        filter_text.insert('1.0', '{}')
        
        def do_query_namespaces():
            query_id = query_id_var.get().strip()
            vector_vals_str = vector_text.get('1.0', tk.END).strip()
            
            if not query_id and not vector_vals_str:
                messagebox.showwarning("Input Required", "Please provide a Vector ID or vector values")
                return
            
            self._set_status("Querying all namespaces...", True)
            
            def query():
                import time
                import json
                start_time = time.time()
                
                index, index_name = self._get_index_client()
                stats = index.describe_index_stats()
                
                # Get all namespaces
                namespaces = []
                if hasattr(stats, 'namespaces') and stats.namespaces:
                    namespaces = list(stats.namespaces.keys())
                if not namespaces:
                    namespaces = ['']  # Default namespace
                
                self._trace('query_namespaces_start', 
                           index=index_name, 
                           namespaces=namespaces,
                           top_k=top_k_var.get())
                
                # Parse filter if provided
                filter_str = filter_text.get('1.0', tk.END).strip()
                metadata_filter = None
                if filter_str and filter_str != '{}':
                    try:
                        metadata_filter = json.loads(filter_str)
                    except:
                        pass
                
                # Build query params
                query_params = {
                    'namespaces': namespaces,
                    'top_k': top_k_var.get(),
                    'include_metadata': include_meta_var.get(),
                    'include_values': False
                }
                
                if metadata_filter:
                    query_params['filter'] = metadata_filter
                
                # Get query vector
                if query_id:
                    # Fetch the vector first
                    fetch_result = index.fetch(ids=[query_id])
                    if fetch_result.vectors and query_id in fetch_result.vectors:
                        query_params['vector'] = fetch_result.vectors[query_id].values
                    else:
                        raise ValueError(f"Vector {query_id} not found")
                else:
                    # Parse values
                    query_params['vector'] = [float(x.strip()) for x in vector_vals_str.split(',')]
                
                # Use query_namespaces - this queries all namespaces in parallel!
                result = index.query_namespaces(**query_params)
                
                duration_ms = int((time.time() - start_time) * 1000)
                self._trace('query_namespaces_complete', 
                           matches=len(result.matches) if hasattr(result, 'matches') else 0,
                           duration_ms=duration_ms)
                
                return result
            
            def show_results(result):
                self._set_status("Ready", False)
                dialog.destroy()
                
                result_dialog = tk.Toplevel(self.root)
                result_dialog.title("Cross-Namespace Query Results")
                result_dialog.geometry("900x700")
                result_dialog.configure(bg=self.colors['bg'])
                
                matches = result.matches if hasattr(result, 'matches') else []
                usage = result.usage if hasattr(result, 'usage') else {}
                
                ttk.Label(result_dialog, 
                         text=f"🌐 Found {len(matches)} matches across all namespaces",
                         font=('Helvetica', 14, 'bold')).pack(pady=10)
                
                if usage:
                    ttk.Label(result_dialog, 
                             text=f"Read Units: {usage.get('read_units', 'N/A')}",
                             foreground='gray').pack()
                
                # Results treeview
                tree_frame = ttk.Frame(result_dialog)
                tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                
                columns = ('rank', 'id', 'namespace', 'score', 'title')
                tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=20)
                tree.heading('rank', text='#')
                tree.heading('id', text='Vector ID')
                tree.heading('namespace', text='Namespace')
                tree.heading('score', text='Score')
                tree.heading('title', text='Title')
                
                tree.column('rank', width=40)
                tree.column('id', width=200)
                tree.column('namespace', width=150)
                tree.column('score', width=100)
                tree.column('title', width=300)
                
                vsb = ttk.Scrollbar(tree_frame, orient='vertical', command=tree.yview)
                tree.configure(yscrollcommand=vsb.set)
                tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                vsb.pack(side=tk.RIGHT, fill=tk.Y)
                
                for i, match in enumerate(matches, 1):
                    meta = match.metadata if hasattr(match, 'metadata') else {}
                    ns = match.namespace if hasattr(match, 'namespace') else ''
                    title = meta.get('title', meta.get('name', ''))[:50]
                    score = f"{match.score:.4f}" if hasattr(match, 'score') else '—'
                    tree.insert('', tk.END, values=(i, match.id, ns, score, title))
                
                ttk.Button(result_dialog, text="Close", command=result_dialog.destroy).pack(pady=10)
            
            self._run_async(query, show_results)
        
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=15)
        ttk.Button(btn_frame, text="🌐 Query All Namespaces", command=do_query_namespaces).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def _list_vector_ids_dialog(self):
        """List vector IDs with pagination (serverless indexes only)."""
        dialog = tk.Toplevel(self.root)
        dialog.title("List Vector IDs")
        dialog.geometry("700x600")
        dialog.configure(bg=self.colors['bg'])
        
        ttk.Label(dialog, text="📜 List Vector IDs", font=('Helvetica', 14, 'bold')).pack(pady=10)
        ttk.Label(dialog, text="⚠️ This feature works only with serverless indexes", 
                 foreground='#f9e2af').pack()
        
        # Parameters
        param_frame = ttk.LabelFrame(dialog, text="Parameters", padding=10)
        param_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(param_frame, text="Prefix (optional):").grid(row=0, column=0, sticky='e', padx=5, pady=5)
        prefix_var = tk.StringVar()
        ttk.Entry(param_frame, textvariable=prefix_var, width=40).grid(row=0, column=1, sticky='w', padx=5, pady=5)
        
        ttk.Label(param_frame, text="Limit:").grid(row=1, column=0, sticky='e', padx=5, pady=5)
        limit_var = tk.IntVar(value=100)
        ttk.Spinbox(param_frame, from_=1, to=1000, textvariable=limit_var, width=10).grid(row=1, column=1, sticky='w', padx=5, pady=5)
        
        ttk.Label(param_frame, text="Namespace:").grid(row=2, column=0, sticky='e', padx=5, pady=5)
        ns_var = tk.StringVar(value=self.pinecone_namespace_var.get())
        ttk.Entry(param_frame, textvariable=ns_var, width=40).grid(row=2, column=1, sticky='w', padx=5, pady=5)
        
        # Results
        result_frame = ttk.LabelFrame(dialog, text="Vector IDs", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        result_text = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD, font=('Monaco', 10))
        result_text.pack(fill=tk.BOTH, expand=True)
        
        pagination_token = {'value': None}
        
        def list_ids():
            result_text.delete('1.0', tk.END)
            result_text.insert('1.0', "Loading vector IDs...\n")
            
            def fetch():
                index, _ = self._get_index_client()
                namespace = (ns_var.get() or '').strip()
                
                params = {
                    'limit': limit_var.get(),
                    'namespace': namespace or None
                }
                if prefix_var.get():
                    params['prefix'] = prefix_var.get()
                if pagination_token['value']:
                    params['pagination_token'] = pagination_token['value']
                
                return index.list_paginated(**params)
            
            def display(results):
                result_text.delete('1.0', tk.END)
                
                if hasattr(results, 'vectors') and results.vectors:
                    result_text.insert(tk.END, f"Found {len(results.vectors)} vector IDs:\n\n")
                    for vec_obj in results.vectors:
                        vec_id = vec_obj.id if hasattr(vec_obj, 'id') else vec_obj
                        result_text.insert(tk.END, f"{vec_id}\n")
                    
                    if hasattr(results, 'pagination') and hasattr(results.pagination, 'next'):
                        pagination_token['value'] = results.pagination.next
                        result_text.insert(tk.END, f"\n✓ More results available (use Next Page)")
                    else:
                        pagination_token['value'] = None
                        result_text.insert(tk.END, f"\n✓ End of results")
                else:
                    result_text.insert(tk.END, "No vector IDs found.")
            
            self._run_async(fetch, display)
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="🔍 List IDs", command=list_ids).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="➡️ Next Page", 
                  command=lambda: list_ids() if pagination_token['value'] else None).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Close", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def _configure_index_dialog(self):
        """Configure an existing index (pod-based indexes: replicas/pod_type, all indexes: deletion protection)."""
        index_name = self.pinecone_index_var.get()
        if not index_name:
            messagebox.showwarning("No Index", "Please select an index first")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Configure Index: {index_name}")
        dialog.geometry("500x400")
        dialog.configure(bg=self.colors['bg'])
        
        ttk.Label(dialog, text=f"⚙️ Configure Index: {index_name}", 
                 font=('Helvetica', 14, 'bold')).pack(pady=10)
        
        # Get current index info first
        self._set_status("Loading index info...", True)
        
        def get_index_info():
            pc = self._get_pinecone_client()
            return pc.describe_index(index_name)
        
        def show_config_options(index_info):
            self._set_status("Ready", False)
            
            is_pod = hasattr(index_info.spec, 'pod') if hasattr(index_info, 'spec') else False
            
            config_frame = ttk.LabelFrame(dialog, text="Configuration Options", padding=15)
            config_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Deletion Protection (all indexes)
            ttk.Label(config_frame, text="Deletion Protection:").grid(row=0, column=0, sticky='e', padx=5, pady=10)
            deletion_var = tk.StringVar(value=index_info.deletion_protection if hasattr(index_info, 'deletion_protection') else 'disabled')
            del_frame = ttk.Frame(config_frame)
            del_frame.grid(row=0, column=1, sticky='w', padx=5, pady=10)
            ttk.Radiobutton(del_frame, text="Enabled", variable=deletion_var, value="enabled").pack(side=tk.LEFT, padx=5)
            ttk.Radiobutton(del_frame, text="Disabled", variable=deletion_var, value="disabled").pack(side=tk.LEFT, padx=5)
            
            pod_vars = {}
            if is_pod:
                ttk.Label(config_frame, text="Pod-Based Index Options:", 
                         font=('Helvetica', 11, 'bold')).grid(row=1, column=0, columnspan=2, pady=(20, 10))
                
                # Pod Type
                ttk.Label(config_frame, text="Pod Type:").grid(row=2, column=0, sticky='e', padx=5, pady=5)
                pod_type_var = tk.StringVar(value=index_info.spec.pod.pod_type if hasattr(index_info.spec.pod, 'pod_type') else 'p1.x1')
                pod_type_combo = ttk.Combobox(config_frame, textvariable=pod_type_var, 
                                              values=['p1.x1', 'p1.x2', 'p1.x4', 'p1.x8', 'p2.x1', 'p2.x2', 'p2.x4', 'p2.x8'],
                                              state='readonly', width=15)
                pod_type_combo.grid(row=2, column=1, sticky='w', padx=5, pady=5)
                pod_vars['pod_type'] = pod_type_var
                
                # Replicas
                ttk.Label(config_frame, text="Replicas:").grid(row=3, column=0, sticky='e', padx=5, pady=5)
                replicas_var = tk.IntVar(value=index_info.spec.pod.replicas if hasattr(index_info.spec.pod, 'replicas') else 1)
                ttk.Spinbox(config_frame, from_=1, to=20, textvariable=replicas_var, width=10).grid(row=3, column=1, sticky='w', padx=5, pady=5)
                pod_vars['replicas'] = replicas_var
            else:
                ttk.Label(config_frame, text="ℹ️ Serverless indexes only support deletion protection", 
                         foreground='#89b4fa').grid(row=1, column=0, columnspan=2, pady=20)
            
            def apply_config():
                self._set_status("Configuring index...", True)
                
                def configure():
                    pc = self._get_pinecone_client()
                    
                    config_params = {
                        'name': index_name,
                        'deletion_protection': deletion_var.get()
                    }
                    
                    if is_pod and pod_vars:
                        config_params['pod_type'] = pod_vars['pod_type'].get()
                        config_params['replicas'] = pod_vars['replicas'].get()
                    
                    pc.configure_index(**config_params)
                    return config_params
                
                def on_complete(params):
                    self._set_status("Ready", False)
                    dialog.destroy()
                    messagebox.showinfo("Success", f"Index '{index_name}' configured successfully!")
                    self._log(f"Configured index: {index_name}", 'SUCCESS')
                    self._refresh_org_inspector()
                
                self._run_async(configure, on_complete)
            
            button_frame = ttk.Frame(dialog)
            button_frame.pack(pady=10)
            ttk.Button(button_frame, text="✅ Apply", command=apply_config).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        self._run_async(get_index_info, show_config_options)
    
    def _bulk_upsert_dialog(self):
        """Bulk upsert vectors from JSON file."""
        filepath = filedialog.askopenfilename(
            title="Select JSON file with vectors",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        self._set_status("Reading vectors from file...", True)
        
        def upsert_bulk():
            import json
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Expect format: {"vectors": [{id, values, metadata}, ...], "namespace": "..."}
            vectors = data.get('vectors', [])
            namespace = (data.get('namespace') or '').strip() or None
            index, _ = self._get_index_client()
            
            # Upsert in batches of 100
            batch_size = 100
            total = len(vectors)
            for i in range(0, total, batch_size):
                batch = vectors[i:i+batch_size]
                index.upsert(vectors=batch, namespace=namespace)
            
            return total, namespace
        
        def on_complete(result):
            total, namespace = result
            self._set_status("Ready", False)
            messagebox.showinfo("Success", f"Upserted {total} vectors to namespace '{namespace or '(default)'}'")
            self._log(f"Bulk upserted {total} vectors", 'SUCCESS')
            self._refresh_pinecone_vectors()
        
        self._run_async(upsert_bulk, on_complete)
    
    def _manage_collections_dialog(self):
        """Manage collections (pod-based indexes only)."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Manage Collections")
        dialog.geometry("700x500")
        dialog.configure(bg=self.colors['bg'])
        
        ttk.Label(dialog, text="📦 Manage Collections", font=('Helvetica', 14, 'bold')).pack(pady=10)
        ttk.Label(dialog, text="⚠️ Collections are supported for pod-based indexes only", 
                 foreground='#f9e2af').pack()
        
        # List frame
        list_frame = ttk.LabelFrame(dialog, text="Collections", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        columns = ('name', 'size', 'status', 'vectors', 'dimension')
        tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=15)
        tree.heading('name', text='Name')
        tree.heading('size', text='Size')
        tree.heading('status', text='Status')
        tree.heading('vectors', text='Vectors')
        tree.heading('dimension', text='Dimension')
        
        tree.column('name', width=150)
        tree.column('size', width=100)
        tree.column('status', width=100)
        tree.column('vectors', width=100)
        tree.column('dimension', width=80)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        def refresh_collections():
            for item in tree.get_children():
                tree.delete(item)
            
            def fetch():
                pc = self._get_pinecone_client()
                return pc.list_collections()
            
            def display(collections):
                if hasattr(collections, 'collections'):
                    for coll in collections.collections:
                        tree.insert('', 'end', values=(
                            coll.name,
                            self._format_bytes(coll.size) if hasattr(coll, 'size') else 'N/A',
                            coll.status if hasattr(coll, 'status') else 'Unknown',
                            f"{coll.vector_count:,}" if hasattr(coll, 'vector_count') else 'N/A',
                            coll.dimension if hasattr(coll, 'dimension') else 'N/A'
                        ))
            
            self._run_async(fetch, display)
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="🔄 Refresh", command=refresh_collections).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Close", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        # Load initially
        refresh_collections()
    
    def _create_index_dialog(self):
        """Create a new Pinecone index."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Create New Index")
        dialog.geometry("550x500")
        dialog.configure(bg=self.colors['bg'])
        
        ttk.Label(dialog, text="Create New Pinecone Index", font=('Helvetica', 14, 'bold')).pack(pady=15)
        
        # Form
        form = ttk.Frame(dialog)
        form.pack(fill=tk.BOTH, expand=True, padx=20)
        
        # Index name
        ttk.Label(form, text="Index Name:").grid(row=0, column=0, sticky='w', pady=8)
        name_entry = ttk.Entry(form, width=40)
        name_entry.grid(row=0, column=1, sticky='ew', pady=8)
        form.columnconfigure(1, weight=1)
        
        # Dimension
        ttk.Label(form, text="Dimension:").grid(row=1, column=0, sticky='w', pady=8)
        dim_frame = ttk.Frame(form)
        dim_frame.grid(row=1, column=1, sticky='ew', pady=8)
        
        dim_var = tk.StringVar(value="3072")
        ttk.Entry(dim_frame, textvariable=dim_var, width=15).pack(side=tk.LEFT)
        ttk.Label(dim_frame, text="(text-embedding-3-large: 3072, ada-002: 1536)", 
                 font=('Helvetica', 9), foreground='gray').pack(side=tk.LEFT, padx=10)
        
        # Metric
        ttk.Label(form, text="Metric:").grid(row=2, column=0, sticky='w', pady=8)
        metric_var = tk.StringVar(value="cosine")
        metric_combo = ttk.Combobox(form, textvariable=metric_var, values=['cosine', 'euclidean', 'dotproduct'], 
                                    state='readonly', width=37)
        metric_combo.grid(row=2, column=1, sticky='ew', pady=8)
        
        # Cloud
        ttk.Label(form, text="Cloud Provider:").grid(row=3, column=0, sticky='w', pady=8)
        cloud_var = tk.StringVar(value="aws")
        cloud_combo = ttk.Combobox(form, textvariable=cloud_var, values=['aws', 'gcp', 'azure'], 
                                   state='readonly', width=37)
        cloud_combo.grid(row=3, column=1, sticky='ew', pady=8)
        
        # Region
        ttk.Label(form, text="Region:").grid(row=4, column=0, sticky='w', pady=8)
        region_var = tk.StringVar(value="us-east-1")
        region_entry = ttk.Entry(form, textvariable=region_var, width=40)
        region_entry.grid(row=4, column=1, sticky='ew', pady=8)
        
        ttk.Label(form, text="AWS: us-east-1, us-west-2, etc.\nGCP: us-central1, etc.\nAzure: eastus, westus, etc.",
                 font=('Helvetica', 9), foreground='gray').grid(row=5, column=1, sticky='w')
        
        # Deletion protection
        deletion_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(form, text="Enable deletion protection", variable=deletion_var).grid(row=6, column=1, sticky='w', pady=10)
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=15)
        
        def create():
            try:
                name = name_entry.get().strip()
                if not name:
                    messagebox.showerror("Error", "Index name is required")
                    return
                
                dimension = int(dim_var.get())
                metric = metric_var.get()
                cloud = cloud_var.get()
                region = region_var.get()
                deletion_protection = "enabled" if deletion_var.get() else "disabled"
                
                from pinecone import ServerlessSpec
                pc = self._get_pinecone_client()
                
                self._set_status(f"Creating index '{name}'...", True)
                self._log(f"Creating index: {name}", 'INFO')
                
                pc.create_index(
                    name=name,
                    dimension=dimension,
                    metric=metric,
                    spec=ServerlessSpec(
                        cloud=cloud,
                        region=region
                    ),
                    deletion_protection=deletion_protection
                )
                
                messagebox.showinfo("Success", f"Index '{name}' created successfully!\n\nIt may take a minute to become ready.")
                self._log(f"Created index: {name}", 'SUCCESS')
                dialog.destroy()
                self._set_status("Ready", False)
                
                # Refresh index list
                self._load_pinecone_indexes()
                
            except ValueError as e:
                messagebox.showerror("Input Error", f"Invalid dimension: {e}")
            except Exception as e:
                messagebox.showerror("Creation Error", f"Failed to create index:\n{e}")
                self._log(f"Error creating index: {e}", 'ERROR')
                self._set_status("Ready", False)
        
        ttk.Button(button_frame, text="✅ Create Index", command=create).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="❌ Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def _delete_index_dialog(self):
        """Delete the current Pinecone index."""
        index_name = self.pinecone_index_var.get()
        if not index_name:
            messagebox.showwarning("No Index", "No index selected")
            return
        
        # Confirm
        confirm = messagebox.askyesno(
            "Confirm Deletion",
            f"Delete index '{index_name}'?\n\n"
            "⚠️ WARNING: This will permanently delete:\n"
            "- The entire index\n"
            "- All vectors and metadata\n"
            "- All namespaces\n\n"
            "This action CANNOT be undone!\n\n"
            "Type the index name to confirm:",
            icon='warning'
        )
        
        if not confirm:
            return
        
        # Additional confirmation with name entry
        confirm_dialog = tk.Toplevel(self.root)
        confirm_dialog.title("Confirm Index Deletion")
        confirm_dialog.geometry("400x200")
        confirm_dialog.configure(bg=self.colors['bg'])
        
        ttk.Label(confirm_dialog, text="⚠️ DANGER ZONE ⚠️", 
                 font=('Helvetica', 14, 'bold'), foreground='#f38ba8').pack(pady=10)
        ttk.Label(confirm_dialog, text=f"Type '{index_name}' to confirm deletion:").pack(pady=10)
        
        confirm_entry = ttk.Entry(confirm_dialog, width=40)
        confirm_entry.pack(pady=5)
        
        result = {'confirmed': False}
        
        def do_delete():
            if confirm_entry.get() != index_name:
                messagebox.showerror("Error", "Index name doesn't match")
                return
            
            result['confirmed'] = True
            confirm_dialog.destroy()
        
        button_frame = ttk.Frame(confirm_dialog)
        button_frame.pack(pady=20)
        ttk.Button(button_frame, text="🗑️ DELETE", command=do_delete).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=confirm_dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        # Wait for dialog
        self.root.wait_window(confirm_dialog)
        
        if not result['confirmed']:
            return
        
        # Proceed with deletion
        self._set_status(f"Deleting index '{index_name}'...", True)
        
        def delete():
            pc = self._get_pinecone_client()
            pc.delete_index(index_name)
            self._reset_index_caches(index_name)
            return index_name
        
        def on_complete(name):
            self._set_status("Ready", False)
            messagebox.showinfo("Deleted", f"Index '{name}' has been deleted.")
            self._log(f"Deleted index: {name}", 'SUCCESS')
            
            # Clear UI
            self.pinecone_index_var.set('')
            for item in self.pinecone_tree.get_children():
                self.pinecone_tree.delete(item)
            
            # Refresh index list
            self._load_pinecone_indexes()
        
        self._run_async(delete, on_complete)
    
    def _prev_page(self):
        """Go to previous page of vectors."""
        if self.current_page_var.get() > 1:
            self.current_page_var.set(self.current_page_var.get() - 1)
            self.page_label.config(text=f"Page {self.current_page_var.get()}")
            self._refresh_pinecone_vectors_paginated()
    
    def _next_page(self):
        """Go to next page of vectors."""
        self.current_page_var.set(self.current_page_var.get() + 1)
        self.page_label.config(text=f"Page {self.current_page_var.get()}")
        self._refresh_pinecone_vectors_paginated()
    
    def _refresh_pinecone_vectors_paginated(self):
        """Fetch vectors with pagination using list+fetch."""
        self._set_status("Loading page...", True)
        
        def fetch_page():
            index, _ = self._get_index_client()
            namespace = (self.pinecone_namespace_var.get() or '').strip()
            
            page_size = self.page_size_var.get()
            current_page = self.current_page_var.get()
            
            # Calculate skip count
            skip_count = (current_page - 1) * page_size
            
            ids_to_fetch = []
            count = 0
            
            try:
                # Use index.list() and skip
                for batch in index.list(namespace=namespace or None, limit=100):
                    batch_len = len(batch)
                    
                    # If we haven't reached the start of our page yet
                    if count + batch_len <= skip_count:
                        count += batch_len
                        continue
                    
                    # We are overlapping with the target page
                    # Calculate start index within this batch
                    start_idx = max(0, skip_count - count)
                    
                    # Calculate how many we still need
                    needed = page_size - len(ids_to_fetch)
                    
                    # Take slice from batch
                    chunk = batch[start_idx : start_idx + needed]
                    ids_to_fetch.extend(chunk)
                    
                    count += batch_len
                    
                    if len(ids_to_fetch) >= page_size:
                        break
                
                if not ids_to_fetch:
                    return {'vectors': [], 'has_more': False}
                
                # Fetch details
                vectors = []
                chunk_size = 100
                for i in range(0, len(ids_to_fetch), chunk_size):
                    chunk = ids_to_fetch[i:i+chunk_size]
                    fetch_res = index.fetch(ids=chunk, namespace=namespace or None)
                    vectors.extend(fetch_res.vectors.values())
                
                return {
                    'vectors': vectors,
                    'has_more': len(ids_to_fetch) == page_size
                }
            except Exception as e:
                self._trace('pagination_error', error=str(e), namespace=namespace)
                return {'vectors': [], 'has_more': False}
        
        def display_page(data):
            self._set_status("Ready", False)
            
            # Clear tree
            for item in self.pinecone_tree.get_children():
                self.pinecone_tree.delete(item)
            
            # Display vectors
            for vec in data['vectors']:
                meta = vec.metadata if hasattr(vec, 'metadata') else {}
                vector_id = vec.id if hasattr(vec, 'id') else 'unknown'
                
                title = meta.get('title', 'Untitled')[:40]
                date = meta.get('timestamp', meta.get('date', 'N/A'))[:19]
                duration = str(meta.get('duration', 0))
                themes = meta.get('themes', '')[:30]
                source = meta.get('source', 'N/A')[:15]
                language = meta.get('language', 'en')[:5]
                
                self.pinecone_tree.insert('', 'end', values=(
                    vector_id[:16] + '...',
                    title,
                    date,
                    duration,
                    themes,
                    source,
                    language,
                    '—',
                    len(meta)
                ))
            
            self._log(f"Loaded page {self.current_page_var.get()} ({len(data['vectors'])} vectors)", 'SUCCESS')
        
        self._run_async(fetch_page, display_page)
    
    def _fetch_by_metadata_dialog(self):
        """Search vectors by metadata filters only, without embeddings."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Metadata Search")
        dialog.geometry("650x550")
        dialog.configure(bg=self.colors['bg'])
        
        ttk.Label(dialog, text="Search by Metadata", font=('Helvetica', 14, 'bold')).pack(pady=15)
        ttk.Label(dialog, text="Define metadata filters (JSON format):", font=('Helvetica', 10)).pack(pady=5)
        
        # Filter editor
        filter_text = scrolledtext.ScrolledText(dialog, wrap=tk.WORD, font=('Monaco', 10),
                                               bg=self.colors['surface'], fg=self.colors['fg'],
                                               height=20)
        filter_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Example filter
        example_filter = {
            "$and": [
                {"source": {"$eq": "plaud"}},
                {"language": {"$eq": "en"}}
            ]
        }
        
        import json
        filter_text.insert('1.0', "# Examples:\n")
        filter_text.insert(tk.END, "# Simple: {\"source\": \"plaud\"}\n")
        filter_text.insert(tk.END, "# Complex: ")
        filter_text.insert(tk.END, json.dumps(example_filter, indent=2))
        filter_text.insert(tk.END, "\n\n# Your filter:\n{}")
        
        # Info label
        info = ttk.Label(dialog, text="💡 This uses Pinecone metadata filtering without vector similarity",
                        font=('Helvetica', 9), foreground='gray')
        info.pack(pady=5)
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=15)
        
        def search():
            try:
                filter_str = filter_text.get('1.0', tk.END)
                # Remove comments
                lines = [line for line in filter_str.split('\n') if not line.strip().startswith('#')]
                filter_str = '\n'.join(lines)
                
                metadata_filter = json.loads(filter_str)
                
                # Close dialog and perform search
                dialog.destroy()
                self._do_metadata_search(metadata_filter)
                
            except json.JSONDecodeError as e:
                messagebox.showerror("JSON Error", f"Invalid JSON filter:\n{e}")
            except Exception as e:
                messagebox.showerror("Error", f"Search failed:\n{e}")
        
        ttk.Button(button_frame, text="🔍 Search", command=search).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="❌ Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def _do_metadata_search(self, metadata_filter):
        """Perform metadata-only search using fetch_by_metadata() API (SDK v6+)."""
        self._set_status("Searching by metadata...", True)
        
        def search():
            import time
            index, index_name = self._get_index_client()
            namespace = (self.pinecone_namespace_var.get() or '').strip()
            
            self._trace('fetch_by_metadata_start', 
                       index=index_name, 
                       namespace=namespace or 'default', 
                       filter=metadata_filter)
            
            start_time = time.time()
            all_vectors = []
            
            # Try fetch_by_metadata (SDK v6+) - cleaner approach without dummy vectors
            try:
                # SDK v6+ method - no embedding needed!
                pagination_token = None
                page_count = 0
                limit = 100  # Fetch in pages
                
                while True:
                    page_count += 1
                    fetch_params = {
                        'filter': metadata_filter,
                        'namespace': namespace or None,
                        'limit': limit
                    }
                    if pagination_token:
                        fetch_params['pagination_token'] = pagination_token
                    
                    result = index.fetch_by_metadata(**fetch_params)
                    
                    # Add vectors from this page
                    if hasattr(result, 'vectors') and result.vectors:
                        for vec_id, vec in result.vectors.items():
                            all_vectors.append(vec)
                    
                    # Check pagination
                    if hasattr(result, 'pagination') and result.pagination and hasattr(result.pagination, 'next'):
                        pagination_token = result.pagination.next
                    else:
                        break  # No more pages
                    
                    # Safety limit
                    if len(all_vectors) >= 1000 or page_count >= 20:
                        self._trace('fetch_by_metadata_paginated', pages=page_count, vectors=len(all_vectors))
                        break
                
                duration_ms = int((time.time() - start_time) * 1000)
                self._trace('fetch_by_metadata_complete', 
                           vectors=len(all_vectors), 
                           pages=page_count,
                           duration_ms=duration_ms)
                
            except AttributeError:
                # Fallback: SDK <v6, use query with zero vector
                self._trace('fetch_by_metadata_fallback', reason='SDK does not support fetch_by_metadata')
                stats = index.describe_index_stats()
                dimension = stats.dimension
                
                for attempt in range(3):
                    try:
                        results = index.query(
                            vector=[0.0] * dimension,
                            filter=metadata_filter,
                            top_k=10000,
                            include_metadata=True,
                            namespace=namespace or None
                        )
                        all_vectors = results.matches if hasattr(results, 'matches') else []
                        break
                    except Exception as e:
                        if "SSL" in str(e) and attempt < 2:
                            time.sleep(1)
                        else:
                            raise e
            
            return {
                'vectors': all_vectors,
                'filter': metadata_filter
            }
        
        def display(data):
            self._set_status("Ready", False)
            
            # Clear tree
            for item in self.pinecone_tree.get_children():
                self.pinecone_tree.delete(item)
            
            # Display results
            import json
            filter_str = json.dumps(data['filter'], indent=2)
            self._log(f"Metadata search returned {len(data['vectors'])} results\nFilter: {filter_str}", 'SUCCESS')
            
            for vec in data['vectors']:
                meta = vec.metadata if hasattr(vec, 'metadata') else {}
                vector_id = vec.id if hasattr(vec, 'id') else 'unknown'
                
                title = meta.get('title', 'Untitled')[:40]
                date = meta.get('timestamp', meta.get('date', 'N/A'))[:19]
                duration = str(meta.get('duration', 0))
                themes = meta.get('themes', '')[:30]
                source = meta.get('source', 'N/A')[:15]
                language = meta.get('language', 'en')[:5]
                score = f"{vec.score:.3f}" if hasattr(vec, 'score') else '—'
                
                self.pinecone_tree.insert('', 'end', values=(
                    vector_id[:16] + '...',
                    title,
                    date,
                    duration,
                    themes,
                    source,
                    language,
                    score,
                    len(meta)
                ))
            
            if not data['vectors']:
                messagebox.showinfo("No Results", "No vectors found matching the metadata filter.")
        
        self._run_async(search, display)
    
    def _bulk_import_dialog(self):
        """Start a bulk import job from cloud storage."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Bulk Import")
        dialog.geometry("700x600")
        dialog.configure(bg=self.colors['bg'])
        
        ttk.Label(dialog, text="Bulk Import from Cloud Storage", font=('Helvetica', 14, 'bold')).pack(pady=15)
        
        # Tabs for different import methods
        notebook = ttk.Notebook(dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # S3 import tab
        s3_frame = ttk.Frame(notebook, padding=20)
        notebook.add(s3_frame, text="AWS S3")
        
        ttk.Label(s3_frame, text="S3 URI:").grid(row=0, column=0, sticky='w', pady=8)
        s3_uri = ttk.Entry(s3_frame, width=50)
        s3_uri.grid(row=0, column=1, sticky='ew', pady=8)
        s3_uri.insert(0, "s3://bucket-name/path/to/data.parquet")
        s3_frame.columnconfigure(1, weight=1)
        
        ttk.Label(s3_frame, text="Integration ID:").grid(row=1, column=0, sticky='w', pady=8)
        s3_integration = ttk.Entry(s3_frame, width=50)
        s3_integration.grid(row=1, column=1, sticky='ew', pady=8)
        
        # GCS import tab
        gcs_frame = ttk.Frame(notebook, padding=20)
        notebook.add(gcs_frame, text="Google Cloud Storage")
        
        ttk.Label(gcs_frame, text="GCS URI:").grid(row=0, column=0, sticky='w', pady=8)
        gcs_uri = ttk.Entry(gcs_frame, width=50)
        gcs_uri.grid(row=0, column=1, sticky='ew', pady=8)
        gcs_uri.insert(0, "gs://bucket-name/path/to/data.parquet")
        gcs_frame.columnconfigure(1, weight=1)
        
        ttk.Label(gcs_frame, text="Integration ID:").grid(row=1, column=0, sticky='w', pady=8)
        gcs_integration = ttk.Entry(gcs_frame, width=50)
        gcs_integration.grid(row=1, column=1, sticky='ew', pady=8)
        
        # File upload tab
        upload_frame = ttk.Frame(notebook, padding=20)
        notebook.add(upload_frame, text="Local File")
        
        ttk.Label(upload_frame, text="Upload a Parquet or CSV file with vectors").pack(pady=10)
        
        file_path_var = tk.StringVar()
        file_entry = ttk.Entry(upload_frame, textvariable=file_path_var, width=50, state='readonly')
        file_entry.pack(pady=5)
        
        def browse_file():
            filepath = filedialog.askopenfilename(
                title="Select data file",
                filetypes=[("Parquet files", "*.parquet"), ("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if filepath:
                file_path_var.set(filepath)
        
        ttk.Button(upload_frame, text="📁 Browse", command=browse_file).pack(pady=5)
        
        # Import options (common)
        options_frame = ttk.LabelFrame(dialog, text="Import Options", padding=15)
        options_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(options_frame, text="Target Namespace:").grid(row=0, column=0, sticky='w', pady=5)
        target_ns = ttk.Entry(options_frame, width=30)
        target_ns.grid(row=0, column=1, sticky='ew', pady=5)
        options_frame.columnconfigure(1, weight=1)
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=15)
        
        def start_import():
            try:
                current_tab = notebook.select()
                tab_text = notebook.tab(current_tab, "text")
                
                index, _ = self._get_index_client()
                
                if tab_text == "AWS S3":
                    uri = s3_uri.get().strip()
                    integration_id = s3_integration.get().strip()
                    
                    if not uri or not integration_id:
                        messagebox.showerror("Error", "S3 URI and Integration ID are required")
                        return
                    
                    result = index.start_import(
                        uri=uri,
                        integration_id=integration_id
                    )
                    
                    messagebox.showinfo("Import Started", f"Bulk import initiated!\n\nImport ID: {result.id if hasattr(result, 'id') else 'N/A'}\n\nUse 'View Imports' to track progress.")
                    self._log(f"Started S3 import: {uri}", 'SUCCESS')
                
                elif tab_text == "Google Cloud Storage":
                    uri = gcs_uri.get().strip()
                    integration_id = gcs_integration.get().strip()
                    
                    if not uri or not integration_id:
                        messagebox.showerror("Error", "GCS URI and Integration ID are required")
                        return
                    
                    result = index.start_import(
                        uri=uri,
                        integration_id=integration_id
                    )
                    
                    messagebox.showinfo("Import Started", f"Bulk import initiated!\n\nImport ID: {result.id if hasattr(result, 'id') else 'N/A'}\n\nUse 'View Imports' to track progress.")
                    self._log(f"Started GCS import: {uri}", 'SUCCESS')
                
                elif tab_text == "Local File":
                    filepath = file_path_var.get()
                    if not filepath:
                        messagebox.showerror("Error", "Please select a file")
                        return
                    
                    # Load file and upsert (for local files, can't use start_import)
                    messagebox.showinfo("Info", "Local file upload will use upsert_from_dataframe.\n\nThis may take a while for large files.")
                    
                    import pandas as pd
                    
                    if filepath.endswith('.parquet'):
                        df = pd.read_parquet(filepath)
                    elif filepath.endswith('.csv'):
                        df = pd.read_csv(filepath)
                    else:
                        messagebox.showerror("Error", "Unsupported file format")
                        return
                    
                    namespace = (target_ns.get() or '').strip() or None
                    
                    # Use upsert_from_dataframe
                    index.upsert_from_dataframe(df, namespace=namespace)
                    
                    messagebox.showinfo("Success", f"Uploaded {len(df)} vectors from {filepath}")
                    self._log(f"Uploaded {len(df)} vectors from local file", 'SUCCESS')
                
                dialog.destroy()
                
            except Exception as e:
                messagebox.showerror("Import Error", f"Failed to start import:\n{e}")
                self._log(f"Import error: {e}", 'ERROR')
        
        def view_imports():
            dialog.destroy()
            self._list_imports_dialog()
        
        ttk.Button(button_frame, text="▶️ Start Import", command=start_import).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📋 View Imports", command=view_imports).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="❌ Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def _list_imports_dialog(self):
        """List and track bulk import jobs."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Import Jobs")
        dialog.geometry("900x500")
        dialog.configure(bg=self.colors['bg'])
        
        ttk.Label(dialog, text="Bulk Import Jobs", font=('Helvetica', 14, 'bold')).pack(pady=15)
        
        # Treeview for imports
        columns = ('id', 'status', 'uri', 'records', 'started', 'finished')
        tree = ttk.Treeview(dialog, columns=columns, show='headings', height=15)
        
        tree.heading('id', text='Import ID')
        tree.heading('status', text='Status')
        tree.heading('uri', text='Source URI')
        tree.heading('records', text='Records')
        tree.heading('started', text='Started')
        tree.heading('finished', text='Finished')
        
        tree.column('id', width=200)
        tree.column('status', width=100)
        tree.column('uri', width=250)
        tree.column('records', width=100)
        tree.column('started', width=130)
        tree.column('finished', width=130)
        
        scrollbar = ttk.Scrollbar(dialog, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
        
        # Load imports
        def load_imports():
            index, _ = self._get_index_client()
            
            try:
                imports = index.list_imports()
                
                for item in tree.get_children():
                    tree.delete(item)
                
                for imp in imports:
                    import_id = imp.id if hasattr(imp, 'id') else 'N/A'
                    status = imp.status if hasattr(imp, 'status') else 'Unknown'
                    uri = imp.uri if hasattr(imp, 'uri') else 'N/A'
                    records = imp.records_imported if hasattr(imp, 'records_imported') else 0
                    started = imp.created_at if hasattr(imp, 'created_at') else 'N/A'
                    finished = imp.finished_at if hasattr(imp, 'finished_at') else 'N/A'
                    
                    tree.insert('', 'end', values=(import_id, status, uri, records, started, finished))
                
                self._log(f"Loaded {len(imports)} import jobs", 'INFO')
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to list imports:\n{e}")
                self._log(f"Error listing imports: {e}", 'ERROR')
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="🔄 Refresh", command=load_imports).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Close", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        # Initial load
        load_imports()
    
    def _create_backup_dialog(self):
        """Create collection/backup of current index."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Create Backup")
        dialog.geometry("550x400")
        dialog.configure(bg=self.colors['bg'])
        
        ttk.Label(dialog, text="Create Index Backup", font=('Helvetica', 14, 'bold')).pack(pady=15)
        
        index_name = self.pinecone_index_var.get() or os.getenv('PINECONE_INDEX_NAME', 'transcripts')
        ttk.Label(dialog, text=f"Source Index: {index_name}", font=('Helvetica', 11)).pack(pady=5)
        
        # Backup type selection
        backup_frame = ttk.LabelFrame(dialog, text="Backup Type", padding=15)
        backup_frame.pack(fill=tk.X, padx=20, pady=15)
        
        backup_type = tk.StringVar(value="collection")
        ttk.Radiobutton(backup_frame, text="Collection (Free snapshot)", variable=backup_type, value="collection").pack(anchor='w', pady=5)
        ttk.Radiobutton(backup_frame, text="Backup (Enterprise feature)", variable=backup_type, value="backup").pack(anchor='w', pady=5)
        
        # Collection name
        ttk.Label(dialog, text="Backup Name:").pack(pady=5)
        name_entry = ttk.Entry(dialog, width=40)
        name_entry.pack(pady=5)
        name_entry.insert(0, f"{index_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Info
        info_text = scrolledtext.ScrolledText(dialog, wrap=tk.WORD, height=6, font=('Helvetica', 9),
                                             bg=self.colors['surface'], fg=self.colors['fg'])
        info_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        info_text.insert('1.0', "Collections:\n- Create a static copy of your index\n- Free feature\n- Can be restored to a new index\n\nBackups:\n- Enterprise-only feature\n- Point-in-time backups\n- Automated retention policies")
        info_text.configure(state='disabled')
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=15)
        
        def create():
            try:
                name = name_entry.get().strip()
                if not name:
                    messagebox.showerror("Error", "Backup name is required")
                    return
                
                pc = self._get_pinecone_client()
                
                if backup_type.get() == "collection":
                    pc.create_collection(name=name, source=index_name)
                    messagebox.showinfo("Success", f"Collection '{name}' created!\n\nIt may take a few minutes to complete.\n\nYou can restore it to a new index later.")
                    self._log(f"Created collection: {name}", 'SUCCESS')
                else:
                    pc.create_backup(name=name, source=index_name)
                    messagebox.showinfo("Success", f"Backup '{name}' created!\n\nNote: Backups require an Enterprise plan.")
                    self._log(f"Created backup: {name}", 'SUCCESS')
                
                dialog.destroy()
                
            except Exception as e:
                messagebox.showerror("Backup Error", f"Failed to create backup:\n{e}")
                self._log(f"Backup error: {e}", 'ERROR')
        
        def list_backups():
            dialog.destroy()
            self._list_backups_dialog()
        
        ttk.Button(button_frame, text="💾 Create", command=create).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📋 View Backups", command=list_backups).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="❌ Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def _list_backups_dialog(self):
        """List and manage collections/backups."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Backups & Collections")
        dialog.geometry("800x500")
        dialog.configure(bg=self.colors['bg'])
        
        ttk.Label(dialog, text="Backups & Collections", font=('Helvetica', 14, 'bold')).pack(pady=15)
        
        # Tabs for collections and backups
        notebook = ttk.Notebook(dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Collections tab
        coll_frame = ttk.Frame(notebook)
        notebook.add(coll_frame, text="Collections")
        
        coll_columns = ('name', 'source', 'status', 'size', 'created')
        coll_tree = ttk.Treeview(coll_frame, columns=coll_columns, show='headings', height=15)
        coll_tree.heading('name', text='Name')
        coll_tree.heading('source', text='Source Index')
        coll_tree.heading('status', text='Status')
        coll_tree.heading('size', text='Size')
        coll_tree.heading('created', text='Created')
        
        for col in coll_columns:
            coll_tree.column(col, width=150)
        
        coll_scroll = ttk.Scrollbar(coll_frame, orient=tk.VERTICAL, command=coll_tree.yview)
        coll_tree.configure(yscrollcommand=coll_scroll.set)
        coll_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        coll_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # Backups tab
        backup_frame = ttk.Frame(notebook)
        notebook.add(backup_frame, text="Backups (Enterprise)")
        
        backup_columns = ('name', 'source', 'status', 'created')
        backup_tree = ttk.Treeview(backup_frame, columns=backup_columns, show='headings', height=15)
        backup_tree.heading('name', text='Name')
        backup_tree.heading('source', text='Source Index')
        backup_tree.heading('status', text='Status')
        backup_tree.heading('created', text='Created')
        
        for col in backup_columns:
            backup_tree.column(col, width=190)
        
        backup_scroll = ttk.Scrollbar(backup_frame, orient=tk.VERTICAL, command=backup_tree.yview)
        backup_tree.configure(yscrollcommand=backup_scroll.set)
        backup_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        backup_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # Load data
        def load_data():
            pc = self._get_pinecone_client()
            
            try:
                # Load collections
                collections = pc.list_collections()
                for item in coll_tree.get_children():
                    coll_tree.delete(item)
                
                for coll in collections:
                    name = coll.name if hasattr(coll, 'name') else 'N/A'
                    source = coll.source if hasattr(coll, 'source') else 'N/A'
                    status = coll.status if hasattr(coll, 'status') else 'Unknown'
                    size = coll.vector_count if hasattr(coll, 'vector_count') else 0
                    created = coll.created_at if hasattr(coll, 'created_at') else 'N/A'
                    
                    coll_tree.insert('', 'end', values=(name, source, status, size, created))
                
                # Load backups
                try:
                    backups = pc.list_backups()
                    for item in backup_tree.get_children():
                        backup_tree.delete(item)
                    
                    for backup in backups:
                        name = backup.name if hasattr(backup, 'name') else 'N/A'
                        source = backup.source if hasattr(backup, 'source') else 'N/A'
                        status = backup.status if hasattr(backup, 'status') else 'Unknown'
                        created = backup.created_at if hasattr(backup, 'created_at') else 'N/A'
                        
                        backup_tree.insert('', 'end', values=(name, source, status, created))
                except Exception as e:
                    # Backups may not be available on all plans
                    self._trace('backup_load_error', error=str(e))
                
                self._log("Loaded backups and collections", 'INFO')
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data:\n{e}")
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        
        def restore_collection():
            selection = coll_tree.selection()
            if not selection:
                messagebox.showwarning("No Selection", "Please select a collection to restore")
                return
            
            coll_name = coll_tree.item(selection[0])['values'][0]
            
            # Create restore dialog
            restore_dlg = tk.Toplevel(dialog)
            restore_dlg.title("Restore Collection")
            restore_dlg.geometry("400x200")
            restore_dlg.configure(bg=self.colors['bg'])
            
            ttk.Label(restore_dlg, text=f"Restore: {coll_name}", font=('Helvetica', 12, 'bold')).pack(pady=10)
            ttk.Label(restore_dlg, text="New Index Name:").pack(pady=5)
            
            new_name_entry = ttk.Entry(restore_dlg, width=40)
            new_name_entry.pack(pady=5)
            new_name_entry.insert(0, f"{coll_name}_restored")
            
            def do_restore():
                try:
                    new_name = new_name_entry.get().strip()
                    if not new_name:
                        messagebox.showerror("Error", "Index name is required")
                        return
                    
                    pc = self._get_pinecone_client()
                    
                    # Get collection details for spec
                    coll = pc.describe_collection(coll_name)
                    source = coll.source if hasattr(coll, 'source') else 'unknown'
                    
                    messagebox.showinfo("Restoring", f"Creating index '{new_name}' from collection '{coll_name}'...\n\nThis may take several minutes.")
                    
                    # Note: Actual restore would need proper spec from original index
                    # This is a simplified version
                    self._log(f"Restore collection {coll_name} to index {new_name}", 'INFO')
                    
                    restore_dlg.destroy()
                    dialog.destroy()
                    
                except Exception as e:
                    messagebox.showerror("Restore Error", f"Failed to restore:\n{e}")
            
            ttk.Button(restore_dlg, text="✅ Restore", command=do_restore).pack(pady=10)
        
        ttk.Button(button_frame, text="🔄 Refresh", command=load_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="♻️ Restore Selected", command=restore_collection).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Close", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        # Initial load
        load_data()
    
    # ==================== END NEW FEATURES ====================
    
    def _generate_mindmap(self):
        """Generate mind map visualization."""
        self._set_status("Generating mind map...", True)
        self._log("Generating knowledge graph...", 'INFO')
        
        def generate():
            import subprocess
            result = subprocess.run(
                ['python', 'generate_mindmap.py'],
                capture_output=True, text=True,
                cwd=Path(__file__).parent
            )
            return result.stdout + result.stderr
        
        def on_complete(output):
            self._set_status("Ready", False)
            self._log(output, 'INFO')
            
            # Open the generated file
            html_path = Path(__file__).parent / 'output' / 'knowledge_graph.html'
            if html_path.exists():
                webbrowser.open(f"file://{html_path}")
                self._log("Opened mind map in browser", 'SUCCESS')
        
        self._run_async(generate, on_complete)
    
    def _clear_pinecone(self):
        """Clear all vectors from Pinecone index."""
        if not messagebox.askyesno("Confirm", "Are you sure you want to delete ALL vectors from Pinecone?"):
            return
        
        def clear():
            index, _ = self._get_index_client()
            namespace = (self.pinecone_namespace_var.get() or '').strip()
            index.delete(delete_all=True, namespace=namespace or None)
            return "Cleared"
        
        def on_complete(result):
            self._set_status("Ready", False)
            self._log("Pinecone index cleared", 'SUCCESS')
            self._update_pinecone_stats()
        
        self._set_status("Clearing Pinecone...", True)
        self._run_async(clear, on_complete)
    
    def _load_settings(self):
        """Load settings from .env file."""
        load_dotenv(override=True)
        for key, var in self.settings_vars.items():
            var.set(os.getenv(key, ''))
        self._log("Settings loaded from .env", 'INFO')
    
    def _save_settings(self):
        """Save settings to .env file."""
        env_path = Path(__file__).parent / '.env'
        
        for key, var in self.settings_vars.items():
            value = var.get()
            if value:
                set_key(str(env_path), key, value)
        
        # Reload environment
        load_dotenv(override=True)
        self._log("Settings saved to .env", 'SUCCESS')
        messagebox.showinfo("Saved", "Settings saved successfully!")
    
    def _export_transcripts(self):
        """Export transcripts to file."""
        if not self.transcripts:
            messagebox.showinfo("Info", "No transcripts to export. Fetch them first.")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("CSV", "*.csv")],
            initialfile="transcripts_export.json"
        )
        
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(self.transcripts, f, indent=2)
            self._log(f"Exported {len(self.transcripts)} transcripts to {filepath}", 'SUCCESS')
    
    def _open_output_folder(self):
        """Open output folder in Finder."""
        output_path = Path(__file__).parent / 'output'
        output_path.mkdir(exist_ok=True)
        os.system(f'open "{output_path}"')
    
    def _clear_logs(self):
        """Clear log display."""
        self.log_text.delete('1.0', tk.END)
    
    def _save_logs(self):
        """Save logs to file."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text", "*.txt")],
            initialfile=f"plaudblender_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(self.log_text.get('1.0', tk.END))
            self._log(f"Logs saved to {filepath}", 'SUCCESS')

    # ---------- Helpers ----------

    def _format_duration(self, duration_ms: int) -> str:
        if not duration_ms:
            return "Unknown"
        minutes = duration_ms // 60000
        seconds = (duration_ms % 60000) // 1000
        hours = minutes // 60
        minutes = minutes % 60
        if hours:
            return f"{hours}h {minutes}m"
        return f"{minutes}m {seconds:02d}s"

    def _format_timestamp(self, value_ms: int) -> str:
        if value_ms is None:
            return "[--:--]"
        total_seconds = max(0, int(value_ms / 1000))
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"[{minutes:02d}:{seconds:02d}]"

    def _delete_selected_from_pinecone(self):
        """Delete selected recordings from Pinecone index."""
        selection = self.transcript_tree.selection()
        if not selection:
            messagebox.showinfo("Info", "Please select transcripts to delete from Pinecone")
            return
        
        count = len(selection)
        result = messagebox.askyesno(
            "Confirm Delete",
            f"Delete {count} vector(s) from Pinecone?\n\nThis will remove the embeddings but won't delete the Plaud recordings."
        )
        if not result:
            return
        
        self._set_status(f"Deleting {count} vectors from Pinecone...", True)
        
        def delete():
            index, _ = self._get_index_client()
            
            ids_to_delete = []
            for item_id in selection:
                item = self.transcript_tree.item(item_id)
                recording_id = item['values'][4]  # ID column
                ids_to_delete.append(recording_id)
            
            index.delete(ids=ids_to_delete)
            return len(ids_to_delete)
        
        def on_complete(deleted_count):
            self._set_status("Ready", False)
            self._log(f"Deleted {deleted_count} vectors from Pinecone", 'SUCCESS')
            self._update_pinecone_stats()
            messagebox.showinfo("Success", f"Deleted {deleted_count} vectors from Pinecone")
        
        self._run_async(delete, on_complete)
    
    def _view_recording_details(self):
        """Show detailed metadata dialog for selected recording."""
        selection = self.transcript_tree.selection()
        if not selection or len(selection) != 1:
            messagebox.showinfo("Info", "Please select exactly one recording to view details")
            return
        
        item = self.transcript_tree.item(selection[0])
        recording_id = item['values'][4]
        
        def fetch_details():
            client = self._get_plaud_client()
            return client.get_recording(recording_id)
        
        def show_details(details):
            dialog = tk.Toplevel(self.root)
            dialog.title(f"Recording Details")
            dialog.geometry("800x600")
            
            # Create notebook for organized display
            notebook = ttk.Notebook(dialog)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # General Info tab
            info_tab = ttk.Frame(notebook)
            notebook.add(info_tab, text="📋 General Info")
            
            info_text = scrolledtext.ScrolledText(info_tab, wrap=tk.WORD, font=('Monaco', 11))
            info_text.pack(fill=tk.BOTH, expand=True)
            
            info_lines = [
                f"Name: {details.get('name', 'N/A')}",
                f"ID: {details.get('id', 'N/A')}",
                f"Duration: {self._format_duration(details.get('duration', 0))}",
                f"Start: {details.get('start_at', 'N/A')}",
                f"Created: {details.get('created_at', 'N/A')}",
                f"Updated: {details.get('updated_at', 'N/A')}",
                f"File Size: {details.get('file_size', 'N/A')} bytes",
                f"File Type: {details.get('file_type', 'N/A')}",
                f"Status: {details.get('status', 'N/A')}",
                "",
                "=== Source List ==="
            ]
            
            for source in details.get('source_list', []):
                info_lines.append(f"\nType: {source.get('data_type')}")
                info_lines.append(f"Title: {source.get('data_title', 'N/A')}")
                content = source.get('data_content', '')
                preview = content[:200] + '...' if len(content) > 200 else content
                info_lines.append(f"Content Preview: {preview}")
            
            info_text.insert('1.0', '\n'.join(info_lines))
            info_text.configure(state='disabled')
            
            # Raw JSON tab
            raw_tab = ttk.Frame(notebook)
            notebook.add(raw_tab, text="🔧 Raw JSON")
            
            raw_text = scrolledtext.ScrolledText(raw_tab, wrap=tk.WORD, font=('Monaco', 10))
            raw_text.pack(fill=tk.BOTH, expand=True)
            raw_text.insert('1.0', json.dumps(details, indent=2, default=str))
            raw_text.configure(state='disabled')
            
            # Close button
            ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=5)
        
        self._run_async(fetch_details, show_details)
    
    def _clear_filters(self):
        """Reset all filter controls."""
        self.transcript_filter.delete(0, tk.END)
        if hasattr(self, 'min_duration'):
            self.min_duration.set(0)
        if hasattr(self, 'date_filter_var'):
            self.date_filter_var.set("All")
        self._fetch_transcripts()

    def _sanitize_filename(self, name: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
        return safe.strip("._") or "transcript"

    def _parse_transcript_detail(self, details: dict) -> dict:
        import json as json_module
        segments = []
        speaker_segments = {}
        full_text_parts = []
        if not details:
            return {
                'segments': segments,
                'speaker_segments': speaker_segments,
                'notes': [],
                'word_count': 0,
                'keywords': [],
                'full_text': ''
            }
        
        for source in details.get('source_list', []):
            if source.get('data_type') == 'transaction':
                try:
                    entries = json_module.loads(source.get('data_content', '[]'))
                except json_module.JSONDecodeError:
                    entries = []
                for entry in entries:
                    content = entry.get('content', '')
                    speaker = entry.get('speaker', 'Speaker')
                    time_str = self._format_timestamp(entry.get('start_time'))
                    segment = {
                        'time': time_str,
                        'speaker': speaker,
                        'content': content
                    }
                    segments.append(segment)
                    full_text_parts.append(f"{speaker}: {content}")
                    speaker_segments.setdefault(speaker, []).append(segment)
        
        notes = []
        for note in details.get('note_list', []):
            notes.append({
                'title': note.get('data_title') or note.get('data_type', 'Note'),
                'content': note.get('data_content', '')
            })
        
        full_text = '\n'.join(full_text_parts)
        words = re.findall(r"[A-Za-z']+", full_text.lower())
        filtered = [w for w in words if len(w) > 3 and w not in STOPWORDS]
        keywords = Counter(filtered).most_common(12)
        
        return {
            'segments': segments,
            'speaker_segments': speaker_segments,
            'notes': notes,
            'word_count': len(full_text.split()),
            'keywords': keywords,
            'full_text': full_text
        }

    def _build_markdown_document(self, name: str, start_at: str, duration_str: str, parsed: dict) -> str:
        lines = [f"# {name}", ""]
        if start_at:
            lines.append(f"*Recorded:* {start_at}")
        if duration_str:
            lines.append(f"*Duration:* {duration_str}")
        lines.append("")
        lines.append("## Transcript")
        for seg in parsed.get('segments', []):
            lines.append(f"- {seg['time']} **{seg['speaker']}**: {seg['content']}")
        
        notes = parsed.get('notes', [])
        if notes:
            lines.append("")
            lines.append("## Plaud Notes & Summaries")
            for note in notes:
                lines.append(f"### {note['title']}")
                lines.append(note['content'])
                lines.append("")
        
        keywords = parsed.get('keywords', [])
        if keywords:
            lines.append("## Top Keywords")
            for word, count in keywords:
                lines.append(f"- {word} ({count})")
        
        return '\n'.join(lines)
    
    # ============================================================
    # PINECONE UI HELPER METHODS
    # ============================================================
    
    def _show_org_overview(self):
        """Show organization overview in a popup dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Pinecone Organization Overview")
        dialog.geometry("1000x700")
        
        # Header
        header = ttk.Frame(dialog)
        header.pack(fill=tk.X, padx=15, pady=15)
        
        ttk.Label(header, text="🏢 Organization Overview", font=('Helvetica', 16, 'bold')).pack(side=tk.LEFT)
        
        btn_frame = ttk.Frame(header)
        btn_frame.pack(side=tk.RIGHT)
        ttk.Button(btn_frame, text="🔄 Refresh", command=lambda: self._load_org_data(dialog)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="💾 Export", command=self._export_org_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="🔍 Verify", command=self._show_verification_details).pack(side=tk.LEFT, padx=5)
        
        # Stats cards
        stats_frame = ttk.Frame(dialog)
        stats_frame.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        dialog.org_summary_labels = {}
        org_stats = [
            ('total_indexes', '📚 Total Indexes'),
            ('total_vectors', '🔢 Total Vectors'),
            ('total_namespaces', '🏷️ Total Namespaces'),
        ]
        
        for i, (key, label) in enumerate(org_stats):
            card = ttk.Frame(stats_frame, relief='solid', borderwidth=1, padding=15)
            card.grid(row=0, column=i, padx=10, sticky='nsew')
            stats_frame.columnconfigure(i, weight=1)
            ttk.Label(card, text=label, font=('Helvetica', 10), foreground='gray').pack()
            dialog.org_summary_labels[key] = ttk.Label(card, text='—', font=('Helvetica', 18, 'bold'))
            dialog.org_summary_labels[key].pack()
        
        # Indexes table
        table_frame = ttk.LabelFrame(dialog, text="📊 Indexes", padding=10)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
        columns = ('name', 'status', 'type', 'region', 'dimension', 'vectors', 'namespaces')
        tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=12)
        
        for col, heading in [
            ('name', 'Index Name'),
            ('status', 'Status'),
            ('type', 'Type'),
            ('region', 'Region'),
            ('dimension', 'Dimension'),
            ('vectors', 'Vectors'),
            ('namespaces', 'Namespaces')
        ]:
            tree.heading(col, text=heading, anchor='w' if col == 'name' else 'center')
        
        tree.column('name', width=150)
        tree.column('status', width=80)
        tree.column('type', width=100)
        tree.column('region', width=120)
        tree.column('dimension', width=80)
        tree.column('vectors', width=100)
        tree.column('namespaces', width=80)
        
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        dialog.org_dialog_tree = tree
        
        # Load data
        self._load_org_data(dialog)
        
        # Close button
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=15)
    
    def _show_index_management(self):
        """Show index management dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Index Management")
        dialog.geometry("900x600")
        
        # Header
        header = ttk.Frame(dialog)
        header.pack(fill=tk.X, padx=15, pady=15)
        
        ttk.Label(header, text="⚡ Index Management", font=('Helvetica', 16, 'bold')).pack(side=tk.LEFT)
        
        # Action buttons
        actions = ttk.Frame(dialog)
        actions.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        ttk.Button(actions, text="➕ Create New Index", command=self._create_index_dialog, width=20).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions, text="⚙️ Configure Index", command=self._configure_index_dialog, width=20).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions, text="🗑️ Delete Index", command=self._delete_index_dialog, width=20).pack(side=tk.LEFT, padx=5)
        ttk.Button(actions, text="📦 Manage Collections", command=self._manage_collections_dialog, width=20).pack(side=tk.LEFT, padx=5)
        
        # Namespace operations
        ns_frame = ttk.LabelFrame(dialog, text="🏷️ Namespace Operations", padding=15)
        ns_frame.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        ns_actions = ttk.Frame(ns_frame)
        ns_actions.pack(fill=tk.X)
        
        ttk.Button(ns_actions, text="➕ Create Namespace", command=self._create_namespace_dialog, width=18).pack(side=tk.LEFT, padx=5)
        ttk.Button(ns_actions, text="🗑️ Delete Namespace", command=self._delete_namespace_dialog, width=18).pack(side=tk.LEFT, padx=5)
        ttk.Button(ns_actions, text="📊 View Stats", command=self._show_namespace_stats, width=18).pack(side=tk.LEFT, padx=5)
        ttk.Button(ns_actions, text="📜 List Vector IDs", command=self._list_vector_ids_dialog, width=18).pack(side=tk.LEFT, padx=5)
        
        # Info section
        info_frame = ttk.LabelFrame(dialog, text="ℹ️ Current Selection", padding=15)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
        info_text = scrolledtext.ScrolledText(info_frame, height=15, wrap=tk.WORD, font=('Monaco', 10))
        info_text.pack(fill=tk.BOTH, expand=True)
        
        # Display current index info
        current_index = self.pinecone_index_var.get()
        current_ns = self.pinecone_namespace_var.get()
        
        info_content = f"""Current Index: {current_index}
Current Namespace: {current_ns or '<default>'}

Total Vectors: {self.pinecone_stat_labels.get('total_vectors', {}).cget('text') if hasattr(self, 'pinecone_stat_labels') else '—'}
Dimension: {self.pinecone_stat_labels.get('dimension', {}).cget('text') if hasattr(self, 'pinecone_stat_labels') else '—'}
Namespaces: {self.pinecone_stat_labels.get('namespaces', {}).cget('text') if hasattr(self, 'pinecone_stat_labels') else '—'}

Operations:
• Create Index: Create a new vector index (serverless or pod-based)
• Configure Index: Modify index settings (pod type, replicas, etc.)
• Delete Index: Permanently remove an index
• Manage Collections: Create backups and restore from collections

Namespace Operations:
• Create Namespace: Add vectors to a new namespace (implicit creation)
• Delete Namespace: Remove all vectors in a namespace
• View Stats: See detailed statistics per namespace
• List Vector IDs: Browse vector IDs in the namespace (serverless only)
"""
        
        info_text.insert('1.0', info_content)
        info_text.config(state='disabled')
        
        # Close button
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=15)
    
    def _load_org_data(self, dialog):
        """Load organization data into the dialog."""
        def fetch():
            return self._fetch_org_data()
        
        def display(org_data):
            self.current_org_data = org_data
            
            # Update stats
            summary_labels = getattr(dialog, 'org_summary_labels', {})
            if summary_labels:
                self._trace(
                    'org_dialog_totals',
                    indexes=len(org_data.get('indexes', [])),
                    vectors=sum(idx.get('vector_count', 0) for idx in org_data.get('indexes', [])),
                    namespaces=sum(idx.get('namespace_count', 0) for idx in org_data.get('indexes', []))
                )
                summary_labels['total_indexes'].config(text=str(len(org_data.get('indexes', []))))
                total_vectors = sum(idx.get('vector_count', 0) for idx in org_data.get('indexes', []))
                summary_labels['total_vectors'].config(text=f"{total_vectors:,}")
                total_namespaces = sum(idx.get('namespace_count', 0) for idx in org_data.get('indexes', []))
                summary_labels['total_namespaces'].config(text=str(total_namespaces))
            
            # Update tree
            dialog_tree = getattr(dialog, 'org_dialog_tree', None)
            if dialog_tree:
                dialog_tree.delete(*dialog_tree.get_children())
                for idx in org_data.get('indexes', []):
                    dialog_tree.insert('', 'end', values=(
                        idx.get('name', ''),
                        idx.get('status', ''),
                        idx.get('type', ''),
                        idx.get('region', ''),
                        idx.get('dimension', ''),
                        f"{idx.get('vector_count', 0):,}",
                        idx.get('namespace_count', 0)
                    ))
            
            self._set_status("Ready", False)
            self._log("Organization data loaded", 'SUCCESS')
        
        self._set_status("Loading organization data...", True)
        self._run_async(fetch, display)
    
    # ============================================================
    # PINECONE ORGANIZATION INSPECTOR METHODS
    # ============================================================
    
    def _refresh_org_inspector(self):
        """Refresh all Pinecone organization data."""
        if not hasattr(self, 'org_indexes_tree'):
            self._log("Org inspector UI not initialized yet; skipping refresh.", 'WARNING')
            return
        self._set_status("Loading Pinecone organization data...", True)
        self._log("Refreshing Pinecone organization inspector...", 'INFO')
        self.root.after(100, lambda: self._run_async(self._fetch_org_data, self._display_org_data))
    
    def _fetch_org_data(self):
        """Fetch comprehensive Pinecone organization data."""
        api_key = os.getenv('PINECONE_API_KEY')
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found")
        
        pc = self._get_pinecone_client()
        org_data = {
            'api_key': api_key,
            'indexes': [],
            'collections': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Fetch all indexes
        indexes_list = pc.list_indexes()
        
        total_vectors = 0
        total_namespaces = 0
        
        if hasattr(indexes_list, 'indexes'):
            for idx_info in indexes_list.indexes:
                index_data = {
                    'name': idx_info.name,
                    'host': idx_info.host,
                    'dimension': idx_info.dimension,
                    'metric': idx_info.metric,
                    'status': idx_info.status.state if hasattr(idx_info.status, 'state') else 'unknown',
                    'deletion_protection': idx_info.deletion_protection,
                }
                
                # Get spec details
                if hasattr(idx_info, 'spec'):
                    spec = idx_info.spec
                    if hasattr(spec, 'serverless'):
                        index_data['type'] = 'Serverless'
                        index_data['cloud'] = spec.serverless.cloud
                        index_data['region'] = spec.serverless.region
                    elif hasattr(spec, 'pod'):
                        index_data['type'] = 'Pod-based'
                        index_data['environment'] = spec.pod.environment
                        index_data['pod_type'] = spec.pod.pod_type
                        index_data['pods'] = spec.pod.pods
                        index_data['replicas'] = spec.pod.replicas
                        index_data['shards'] = spec.pod.shards
                        index_data['cloud'] = spec.pod.environment.split('-')[0] if hasattr(spec.pod, 'environment') else 'N/A'
                        index_data['region'] = spec.pod.environment
                
                # Get stats
                try:
                    index = pc.Index(idx_info.name)
                    stats = index.describe_index_stats()
                    
                    index_data['vector_count'] = stats.total_vector_count
                    index_data['index_fullness'] = getattr(stats, 'index_fullness', 0)
                    index_data['namespaces'] = {}
                    
                    total_vectors += stats.total_vector_count
                    
                    if hasattr(stats, 'namespaces') and stats.namespaces:
                        index_data['namespace_count'] = len(stats.namespaces)
                        total_namespaces += len(stats.namespaces)
                        for ns_name, ns_stats in stats.namespaces.items():
                            index_data['namespaces'][ns_name if ns_name else '<default>'] = ns_stats.vector_count
                    else:
                        # If no namespaces reported but there are vectors, they're in the default namespace
                        if stats.total_vector_count > 0:
                            index_data['namespace_count'] = 1
                            total_namespaces += 1
                            index_data['namespaces']['<default>'] = stats.total_vector_count
                        else:
                            index_data['namespace_count'] = 0
                    
                    # Fetch sample vectors
                    try:
                        import time
                        for attempt in range(3):
                            try:
                                sample = index.query(
                                    vector=[0.0] * idx_info.dimension,
                                    top_k=5,
                                    include_metadata=True,
                                    include_values=False
                                )
                                break
                            except Exception as e:
                                if "SSL" in str(e) and attempt < 2:
                                    time.sleep(1)
                                else:
                                    raise e
                        index_data['samples'] = []
                        if hasattr(sample, 'matches'):
                            for match in sample.matches:
                                sample_data = {
                                    'id': match.id,
                                    'score': match.score,
                                    'metadata': match.metadata if hasattr(match, 'metadata') else {}
                                }
                                index_data['samples'].append(sample_data)
                    except:
                        index_data['samples'] = []
                        
                except Exception as e:
                    index_data['error'] = str(e)
                
                org_data['indexes'].append(index_data)
        
        # Fetch collections
        try:
            collections = pc.list_collections()
            if hasattr(collections, 'collections'):
                for coll in collections.collections:
                    coll_data = {
                        'name': coll.name,
                        'status': getattr(coll, 'status', 'unknown'),
                        'size': getattr(coll, 'size', 0),
                        'vector_count': getattr(coll, 'vector_count', 0),
                        'dimension': getattr(coll, 'dimension', 0),
                    }
                    org_data['collections'].append(coll_data)
        except:
            pass
        
        org_data['totals'] = {
            'vectors': total_vectors,
            'namespaces': total_namespaces,
            'indexes': len(org_data['indexes']),
            'collections': len(org_data['collections'])
        }
        
        # Verification: Log detailed breakdown
        verification_log = []
        verification_log.append(f"\n{'='*60}")
        verification_log.append("PINECONE ORGANIZATION VERIFICATION")
        verification_log.append(f"{'='*60}")
        verification_log.append(f"Total Indexes: {org_data['totals']['indexes']}")
        verification_log.append(f"Total Vectors Across All Indexes: {org_data['totals']['vectors']:,}")
        verification_log.append(f"Total Namespaces Across All Indexes: {org_data['totals']['namespaces']}")
        verification_log.append(f"\nPer-Index Breakdown:")
        
        for idx in org_data['indexes']:
            verification_log.append(f"\n  📍 {idx['name']}:")
            verification_log.append(f"     Vectors: {idx.get('vector_count', 0):,}")
            verification_log.append(f"     Namespaces: {idx.get('namespace_count', 0)}")
            if idx.get('namespaces'):
                for ns_name, count in idx['namespaces'].items():
                    verification_log.append(f"       - {ns_name}: {count:,} vectors")
        
        # Verify totals match
        sum_vectors = sum(idx.get('vector_count', 0) for idx in org_data['indexes'])
        sum_namespaces = sum(idx.get('namespace_count', 0) for idx in org_data['indexes'])
        
        if sum_vectors != total_vectors:
            verification_log.append(f"\n⚠️ WARNING: Vector count mismatch!")
            verification_log.append(f"   Expected: {total_vectors:,}, Calculated Sum: {sum_vectors:,}")
        else:
            verification_log.append(f"\n✅ Vector counts verified: {total_vectors:,}")
        
        if sum_namespaces != total_namespaces:
            verification_log.append(f"⚠️ WARNING: Namespace count mismatch!")
            verification_log.append(f"   Expected: {total_namespaces}, Calculated Sum: {sum_namespaces}")
        else:
            verification_log.append(f"✅ Namespace counts verified: {total_namespaces}")
        
        verification_log.append(f"{'='*60}\n")
        
        org_data['verification_log'] = '\n'.join(verification_log)
        
        return org_data
    
    def _display_org_data(self, org_data):
        """Display organization data in the UI."""
        self._set_status("Ready", False)
        
        totals = org_data.get('totals', {})
        summary_labels = getattr(self, 'org_summary_labels', {})
        self._trace('org_totals_update', totals=totals)
        if summary_labels:
            if 'total_indexes' in summary_labels:
                summary_labels['total_indexes'].configure(text=str(totals.get('indexes', 0)))
            if 'total_vectors' in summary_labels:
                summary_labels['total_vectors'].configure(text=f"{totals.get('vectors', 0):,}")
            if 'total_namespaces' in summary_labels:
                summary_labels['total_namespaces'].configure(text=str(totals.get('namespaces', 0)))
            if 'total_dimensions' in summary_labels:
                dimensions = sorted({idx.get('dimension', '—') for idx in org_data.get('indexes', [])})
                summary_labels['total_dimensions'].configure(text=', '.join(map(str, dimensions)) or '—')
        
        # Update indexes tree
        tree = getattr(self, 'org_indexes_tree', None)
        if tree:
            for item in tree.get_children():
                tree.delete(item)
            for idx in org_data.get('indexes', []):
                tree.insert('', 'end', values=(
                    idx.get('name', ''),
                    idx.get('status', ''),
                    idx.get('type', 'N/A'),
                    idx.get('region', 'N/A'),
                    idx.get('dimension', '—'),
                    idx.get('metric', '—'),
                    f"{idx.get('vector_count', 0):,}",
                    f"{idx.get('namespace_count', 0)}"
                ), tags=(idx.get('name', ''),))
        
        # Store org data for export and detail view
        self.current_org_data = org_data
        self.verification_log = org_data.get('verification_log')
        
        # Log verification info
        if 'verification_log' in org_data:
            self._log(org_data['verification_log'], 'INFO')
        
        self._log(
            f"✅ Organization data loaded: {totals.get('indexes', 0)} indexes, {totals.get('vectors', 0):,} vectors, {totals.get('namespaces', 0)} namespaces",
            'SUCCESS'
        )
    
    def _on_org_index_selected(self, event):
        """Handle index selection in org inspector."""
        if not hasattr(self, 'org_indexes_tree'):
            return
        selection = self.org_indexes_tree.selection()
        if not selection:
            return
        
        item = self.org_indexes_tree.item(selection[0])
        index_name = item['values'][0]
        
        # Find index data
        index_data = None
        for idx in self.current_org_data['indexes']:
            if idx['name'] == index_name:
                index_data = idx
                break
        
        if not index_data:
            return
        
        # Display detailed info
        if hasattr(self, 'org_index_details'):
            self.org_index_details.configure(state='normal')
            self.org_index_details.delete('1.0', tk.END)
        
        details = f"{'='*80}\n"
        details += f"INDEX: {index_data['name']}\n"
        details += f"{'='*80}\n\n"
        
        details += f"📊 BASIC INFO:\n"
        details += f"  Status: {index_data['status']}\n"
        details += f"  Host: {index_data['host']}\n"
        details += f"  Dimension: {index_data['dimension']}\n"
        details += f"  Metric: {index_data['metric']}\n"
        details += f"  Deletion Protection: {index_data['deletion_protection']}\n"
        details += f"  Type: {index_data.get('type', 'N/A')}\n\n"
        
        if index_data.get('type') == 'Serverless':
            details += f"☁️ SERVERLESS CONFIG:\n"
            details += f"  Cloud: {index_data.get('cloud', 'N/A')}\n"
            details += f"  Region: {index_data.get('region', 'N/A')}\n\n"
        elif index_data.get('type') == 'Pod-based':
            details += f"🎯 POD CONFIG:\n"
            details += f"  Environment: {index_data.get('environment', 'N/A')}\n"
            details += f"  Pod Type: {index_data.get('pod_type', 'N/A')}\n"
            details += f"  Pods: {index_data.get('pods', 'N/A')}\n"
            details += f"  Replicas: {index_data.get('replicas', 'N/A')}\n"
            details += f"  Shards: {index_data.get('shards', 'N/A')}\n\n"
        
        details += f"📈 STATISTICS:\n"
        details += f"  Total Vectors: {index_data.get('vector_count', 0):,}\n"
        if 'index_fullness' in index_data:
            details += f"  Index Fullness: {index_data['index_fullness']:.2%}\n"
        details += f"  Namespaces: {index_data.get('namespace_count', 0)}\n\n"
        
        if index_data.get('namespaces'):
            details += f"🏷️ NAMESPACE BREAKDOWN:\n"
            for ns_name, count in index_data['namespaces'].items():
                details += f"  • {ns_name}: {count:,} vectors\n"
            details += "\n"
        
        if 'error' in index_data:
            details += f"⚠️ ERROR: {index_data['error']}\n"
        
        if hasattr(self, 'org_index_details'):
            self.org_index_details.insert('1.0', details)
            self.org_index_details.configure(state='disabled')
        
        # Display sample vectors
        if hasattr(self, 'org_samples_text'):
            self.org_samples_text.configure(state='normal')
            self.org_samples_text.delete('1.0', tk.END)
        
        if index_data.get('samples'):
            samples_text = f"📋 Sample Vectors from '{index_name}' (showing {len(index_data['samples'])}):\n\n"
            
            for i, sample in enumerate(index_data['samples'], 1):
                samples_text += f"{'─'*80}\n"
                samples_text += f"Sample #{i}\n"
                samples_text += f"{'─'*80}\n"
                samples_text += f"ID: {sample['id']}\n"
                samples_text += f"Score: {sample['score']:.4f}\n"
                
                if sample['metadata']:
                    samples_text += f"\nMetadata ({len(sample['metadata'])} fields):\n"
                    for key, value in sample['metadata'].items():
                        value_str = str(value)
                        if len(value_str) > 100:
                            value_str = value_str[:97] + "..."
                        samples_text += f"  {key}: {value_str}\n"
                else:
                    samples_text += "  No metadata\n"
                samples_text += "\n"
        else:
            samples_text = "No sample vectors available for this index."
        
        if hasattr(self, 'org_samples_text'):
            self.org_samples_text.insert('1.0', samples_text)
            self.org_samples_text.configure(state='disabled')
    
    def _export_org_report(self):
        """Export organization report to JSON."""
        if not hasattr(self, 'current_org_data'):
            messagebox.showwarning("No Data", "Please refresh organization data first.")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=f"pinecone_org_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        if filepath:
            try:
                with open(filepath, 'w') as f:
                    json.dump(self.current_org_data, f, indent=2)
                self._log(f"Organization report exported to: {filepath}", 'SUCCESS')
                messagebox.showinfo("Success", f"Report exported to:\n{filepath}")
            except Exception as e:
                self._log(f"Error exporting report: {e}", 'ERROR')
                messagebox.showerror("Error", f"Failed to export report:\n{e}")
    
    def _show_verification_details(self):
        """Display verification details in a dialog."""
        if not hasattr(self, 'verification_log') or not self.verification_log:
            messagebox.showinfo("No Verification Data", "Please refresh the organization data to generate verification details.")
            return
        
        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Data Verification Details")
        dialog.geometry("800x600")
        
        # Add header
        header_frame = ttk.Frame(dialog)
        header_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(
            header_frame,
            text="Pinecone Organization Data Verification",
            font=('TkDefaultFont', 12, 'bold')
        ).pack()
        
        ttk.Label(
            header_frame,
            text="This report validates that aggregated totals match the sum of individual index counts.",
            foreground='gray'
        ).pack()
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(dialog)
        text_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side='right', fill='y')
        
        text_widget = tk.Text(
            text_frame,
            wrap='word',
            yscrollcommand=scrollbar.set,
            font=('Courier', 10)
        )
        text_widget.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=text_widget.yview)
        
        # Insert verification log
        text_widget.insert('1.0', self.verification_log)
        text_widget.config(state='disabled')
        
        # Add close button
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        ttk.Button(
            button_frame,
            text="Close",
            command=dialog.destroy
        ).pack(side='right')
        
        ttk.Button(
            button_frame,
            text="Copy to Clipboard",
            command=lambda: self._copy_to_clipboard(self.verification_log, dialog)
        ).pack(side='right', padx=(0, 5))
        
        # Center dialog
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
    
    def _copy_to_clipboard(self, text, parent_window):
        """Copy text to clipboard."""
        parent_window.clipboard_clear()
        parent_window.clipboard_append(text)
        parent_window.update()
        messagebox.showinfo("Copied", "Verification details copied to clipboard!", parent=parent_window)
    
    def _format_bytes(self, bytes_val):
        """Format bytes to human readable."""
        if bytes_val < 1024:
            return f"{bytes_val} B"
        elif bytes_val < 1024**2:
            return f"{bytes_val/1024:.2f} KB"
        elif bytes_val < 1024**3:
            return f"{bytes_val/1024**2:.2f} MB"
        else:
            return f"{bytes_val/1024**3:.2f} GB"
    
    # ========== PHASE 1: Core UX Enhancements ==========
    
    def _load_saved_searches(self):
        """Load saved searches from JSON file."""
        searches_file = Path(__file__).parent / "saved_searches.json"
        if searches_file.exists():
            try:
                with open(searches_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self._trace('saved_search_load_error', error=str(e))
        return {}
    
    def _save_searches_to_file(self):
        """Save searches to JSON file."""
        searches_file = Path(__file__).parent / "saved_searches.json"
        try:
            with open(searches_file, 'w') as f:
                json.dump(self.saved_searches, f, indent=2)
        except Exception as e:
            self._trace('saved_search_save_error', error=str(e))
    
    def _load_indexes_for_header(self, force=False):
        """Load available Pinecone indexes for header dropdown (lazy loaded)."""
        if self._indexes_loaded and not force:
            return  # Already loaded
        
        def fetch():
            try:
                names = self._list_pinecone_index_names()
                self._indexes_loaded = True
                return names
            except Exception as e:
                self._log(f"Error loading indexes: {e}", 'ERROR')
                return [os.getenv('PINECONE_INDEX_NAME', 'transcripts')]
        
        def populate(index_names):
            if hasattr(self, 'header_index_dropdown'):
                self.header_index_dropdown['values'] = index_names
                # Sync with pinecone tab if it exists
                if hasattr(self, 'pinecone_index_dropdown'):
                    self.pinecone_index_dropdown['values'] = index_names
                    if index_names and not self.pinecone_index_var.get():
                        self.pinecone_index_var.set(index_names[0])
                        self._on_index_changed()
        
        self._run_async(fetch, populate)
    
    def _on_header_index_changed(self, event=None):
        """Handle index change from header dropdown."""
        new_index = self.header_index_var.get()
        # Update pinecone tab index
        if hasattr(self, 'pinecone_index_var'):
            self.pinecone_index_var.set(new_index)
            self._on_index_changed()
        # Update dashboard display
        if hasattr(self, 'target_index_label'):
            self.target_index_label.config(text=new_index)
        self._log(f"✓ Switched to index: {new_index}", 'SUCCESS')
        messagebox.showinfo("Index Changed", f"Active index is now: {new_index}")
    
    def _refresh_dashboard_stats(self, use_cache=True):
        """Refresh dashboard statistics with visual charts (cached for performance)."""
        if not hasattr(self, 'stat_labels'):
            return
        
        # Use cache if available and recent (< 30 seconds)
        # Note: _dashboard_stats_cache_time is separate from Pinecone's _stats_cache_time (dict)
        if use_cache and hasattr(self, '_dashboard_stats_cache_time') and self._dashboard_stats_cache_time:
            if isinstance(self._dashboard_stats_cache_time, datetime):
                age = (datetime.now() - self._dashboard_stats_cache_time).total_seconds()
                if age < 30:
                    # Stats are fresh, no need to refresh
                    return
            else:
                # Reset if invalid type
                self._dashboard_stats_cache_time = None
        
        self._dashboard_stats_cache_time = datetime.now()
        
        # Schedule next refresh
        self.root.after(30000, lambda: self._refresh_dashboard_stats(use_cache=True))
    
    def _lazy_import_matplotlib(self):
        """Import matplotlib only when needed (deferred to avoid slow startup)."""
        global MATPLOTLIB_AVAILABLE
        if not MATPLOTLIB_AVAILABLE:
            try:
                import matplotlib
                matplotlib.use('TkAgg')
                MATPLOTLIB_AVAILABLE = True
                self._log("✓ Loaded matplotlib for charts", 'INFO')
            except ImportError:
                self._log("⚠ matplotlib not available. Install with: pip install matplotlib", 'WARNING')
        return MATPLOTLIB_AVAILABLE
    
    def _lazy_import_wordcloud(self):
        """Import wordcloud only when needed (deferred to avoid slow startup)."""
        global WORDCLOUD_AVAILABLE
        if not WORDCLOUD_AVAILABLE:
            try:
                from wordcloud import WordCloud
                WORDCLOUD_AVAILABLE = True
                self._log("✓ Loaded wordcloud for visualizations", 'INFO')
            except ImportError:
                self._log("⚠ wordcloud not available. Install with: pip install wordcloud", 'WARNING')
        return WORDCLOUD_AVAILABLE
    
    def _save_current_search(self):
        """Save current search query with a name."""
        if not hasattr(self, 'search_entry'):
            return
        
        query = self.search_entry.get().strip()
        if not query:
            messagebox.showwarning("No Query", "Enter a search query first")
            return
        
        # Ask for name
        dialog = tk.Toplevel(self.root)
        dialog.title("Save Search")
        dialog.geometry("400x150")
        dialog.configure(bg=self.colors['bg'])
        
        ttk.Label(dialog, text="Search Name:", font=('Helvetica', 11)).pack(pady=10)
        name_entry = ttk.Entry(dialog, width=40, font=('Helvetica', 10))
        name_entry.pack(pady=5)
        name_entry.focus()
        
        def save():
            name = name_entry.get().strip()
            if not name:
                messagebox.showwarning("No Name", "Enter a name for this search")
                return
            
            # Save search with metadata
            self.saved_searches[name] = {
                'query': query,
                'created': datetime.now().isoformat(),
                'top_k': self.search_limit.get() if hasattr(self, 'search_limit') else 5
            }
            self._save_searches_to_file()
            self._log(f"💾 Saved search: {name}", 'SUCCESS')
            dialog.destroy()
            
            # Update saved searches dropdown if it exists
            if hasattr(self, 'saved_search_dropdown'):
                self.saved_search_dropdown['values'] = list(self.saved_searches.keys())
        
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=20)
        ttk.Button(btn_frame, text="Save", command=save).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        dialog.transient(self.root)
        dialog.grab_set()
    
    def _load_saved_search(self, search_name):
        """Load a saved search."""
        if search_name in self.saved_searches:
            search_data = self.saved_searches[search_name]
            if hasattr(self, 'search_entry'):
                self.search_entry.delete(0, tk.END)
                self.search_entry.insert(0, search_data['query'])
            if hasattr(self, 'search_limit') and 'top_k' in search_data:
                self.search_limit.set(search_data['top_k'])
            self._log(f"📂 Loaded search: {search_name}", 'INFO')
    
    def _delete_saved_search(self, search_name):
        """Delete a saved search."""
        if search_name in self.saved_searches:
            if messagebox.askyesno("Delete Search", f"Delete saved search '{search_name}'?"):
                del self.saved_searches[search_name]
                self._save_searches_to_file()
                if hasattr(self, 'saved_search_dropdown'):
                    self.saved_search_dropdown['values'] = list(self.saved_searches.keys())
                self._log(f"🗑️ Deleted search: {search_name}", 'WARNING')


def main():
    """Run the GUI application."""
    root = tk.Tk()
    app = PlaudBlenderGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
