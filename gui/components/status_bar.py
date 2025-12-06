import tkinter as tk
from tkinter import ttk
from gui.state import state


class StatusBar(ttk.Frame):
    """
    Enhanced status bar with real-time metrics.
    
    Displays: status message, latency (ms), Pinecone read units,
    and busy indicator for operation visibility.
    """
    
    def __init__(self, parent):
        super().__init__(parent, style="Panel.TFrame", padding=(6, 3))
        
        # Status message (left side)
        self.message_var = tk.StringVar(value=state.status_message)
        ttk.Label(self, textvariable=self.message_var, style="Muted.TLabel").pack(side=tk.LEFT)
        
        # Metrics container (right side)
        metrics_frame = ttk.Frame(self, style="Panel.TFrame")
        metrics_frame.pack(side=tk.RIGHT)
        
        # Busy indicator
        self.busy_var = tk.StringVar(value="")
        ttk.Label(metrics_frame, textvariable=self.busy_var, 
                  style="Muted.TLabel", width=2).pack(side=tk.RIGHT, padx=(4, 0))
        
        # Latency metric
        self.latency_var = tk.StringVar(value="")
        self.latency_label = ttk.Label(metrics_frame, textvariable=self.latency_var, 
                                        style="Muted.TLabel", width=10)
        self.latency_label.pack(side=tk.RIGHT, padx=(8, 0))
        
        # Read units metric
        self.read_units_var = tk.StringVar(value="")
        self.read_units_label = ttk.Label(metrics_frame, textvariable=self.read_units_var,
                                           style="Muted.TLabel", width=10)
        self.read_units_label.pack(side=tk.RIGHT, padx=(8, 0))
        
        # Namespace indicator
        self.namespace_var = tk.StringVar(value="")
        ttk.Label(metrics_frame, textvariable=self.namespace_var,
                  style="Muted.TLabel").pack(side=tk.RIGHT, padx=(8, 0))

    def update_status(self):
        """Refresh status bar from global state."""
        self.message_var.set(state.status_message)
        self.busy_var.set("●" if state.is_busy else "")
        
        # Update metrics if available in state
        if hasattr(state, 'last_latency_ms') and state.last_latency_ms is not None:
            self.latency_var.set(f"⏱{state.last_latency_ms:.0f}ms")
        else:
            self.latency_var.set("")
            
        if hasattr(state, 'last_read_units') and state.last_read_units is not None:
            self.read_units_var.set(f"📊{state.last_read_units}RU")
        else:
            self.read_units_var.set("")
            
        if hasattr(state, 'active_namespace') and state.active_namespace:
            self.namespace_var.set(f"[{state.active_namespace}]")
        else:
            self.namespace_var.set("")
    
    def set_metrics(self, latency_ms: float = None, read_units: int = None, namespace: str = None):
        """
        Directly set metrics without going through state.
        
        Args:
            latency_ms: Query latency in milliseconds
            read_units: Pinecone read units consumed
            namespace: Active namespace name
        """
        if latency_ms is not None:
            self.latency_var.set(f"⏱{latency_ms:.0f}ms")
            state.last_latency_ms = latency_ms
        if read_units is not None:
            self.read_units_var.set(f"📊{read_units}RU")
            state.last_read_units = read_units
        if namespace is not None:
            self.namespace_var.set(f"[{namespace}]")
            state.active_namespace = namespace
