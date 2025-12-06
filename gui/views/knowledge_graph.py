"""
Knowledge Graph Visualization View

Interactive vis.js-based knowledge graph showing:
- Entities extracted from transcripts (people, places, concepts, organizations)
- Relationships between entities
- Connections to original transcripts
- Clickable nodes with metadata tooltips

Uses the existing vis.js library in lib/vis-9.1.2/
"""
import tkinter as tk
from tkinter import ttk
import json
import os
import tempfile
import webbrowser
from typing import List, Dict, Optional
from gui.views.base import BaseView
from gui.utils.tooltips import ToolTip


class KnowledgeGraphView(BaseView):
    """
    Interactive knowledge graph visualization using vis.js.
    
    Shows entities and relationships extracted via GraphRAG,
    with filtering by entity type and confidence threshold.
    """
    
    def _build(self):
        # Header
        hero = ttk.Frame(self, style="Main.TFrame")
        hero.pack(fill="x", pady=(0, 8))
        ttk.Label(hero, text="Knowledge Graph", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            hero, 
            text="Entity-relationship graph extracted from your transcripts via GraphRAG",
            style="Muted.TLabel"
        ).pack(anchor="w")
        
        # Stats bar
        stats_frame = ttk.Frame(self, style="Panel.TFrame")
        stats_frame.pack(fill="x", pady=(0, 8))
        
        self.stats_labels = {
            'entities': ttk.Label(stats_frame, text="Entities: 0", style="Muted.TLabel"),
            'relationships': ttk.Label(stats_frame, text="Relationships: 0", style="Muted.TLabel"),
            'transcripts': ttk.Label(stats_frame, text="Transcripts: 0", style="Muted.TLabel"),
            'confidence': ttk.Label(stats_frame, text="Avg Confidence: —", style="Muted.TLabel"),
        }
        for label in self.stats_labels.values():
            label.pack(side="left", padx=(0, 16))
        
        # Controls frame
        controls = ttk.LabelFrame(self, text="Graph Controls", padding=8, style="Panel.TLabelframe")
        controls.pack(fill="x", pady=(0, 8))
        
        # Row 1: Entity type filters
        filter_row = ttk.Frame(controls, style="Panel.TFrame")
        filter_row.pack(fill="x", pady=(0, 4))
        
        ttk.Label(filter_row, text="Entity Types:", style="Muted.TLabel").pack(side="left", padx=(0, 8))
        
        self.entity_filters = {}
        entity_types = ['person', 'organization', 'location', 'concept', 'event', 'product']
        for etype in entity_types:
            var = tk.BooleanVar(value=True)
            self.entity_filters[etype] = var
            cb = ttk.Checkbutton(
                filter_row, 
                text=etype.capitalize(), 
                variable=var,
                command=self._on_filter_change
            )
            cb.pack(side="left", padx=(0, 8))
        
        # Row 2: Confidence threshold + layout options
        options_row = ttk.Frame(controls, style="Panel.TFrame")
        options_row.pack(fill="x", pady=(4, 0))
        
        # Confidence threshold slider
        ttk.Label(options_row, text="Min Confidence:", style="Muted.TLabel").pack(side="left", padx=(0, 4))
        self.confidence_var = tk.DoubleVar(value=0.5)
        self.confidence_scale = ttk.Scale(
            options_row,
            from_=0.0,
            to=1.0,
            orient="horizontal",
            variable=self.confidence_var,
            command=self._on_confidence_change,
            length=150
        )
        self.confidence_scale.pack(side="left", padx=(0, 4))
        ToolTip(self.confidence_scale, "Filter entities by confidence score (0.0 = all, 1.0 = highest only)")
        
        self.confidence_label = ttk.Label(options_row, text="0.50", style="Muted.TLabel", width=5)
        self.confidence_label.pack(side="left", padx=(0, 16))
        
        # Layout selection
        ttk.Label(options_row, text="Layout:", style="Muted.TLabel").pack(side="left", padx=(0, 4))
        self.layout_var = tk.StringVar(value="forceAtlas2Based")
        layout_combo = ttk.Combobox(
            options_row,
            textvariable=self.layout_var,
            values=["forceAtlas2Based", "barnesHut", "hierarchicalRepulsion", "repulsion"],
            state="readonly",
            width=18
        )
        layout_combo.pack(side="left", padx=(0, 8))
        layout_combo.bind("<<ComboboxSelected>>", self._on_layout_change)
        ToolTip(layout_combo, "Graph physics simulation algorithm")
        
        # Show labels toggle
        self.show_labels_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_row,
            text="Show Labels",
            variable=self.show_labels_var,
            command=self._refresh_graph
        ).pack(side="left", padx=(8, 0))
        
        # Action buttons
        btn_frame = ttk.Frame(controls, style="Panel.TFrame")
        btn_frame.pack(fill="x", pady=(8, 0))
        
        ttk.Button(
            btn_frame,
            text="🔄 Refresh Graph",
            command=lambda: self.call('refresh_knowledge_graph'),
            style="Accent.TButton"
        ).pack(side="left", padx=(0, 8))
        ToolTip(btn_frame.winfo_children()[-1], "Re-extract entities from all transcripts")
        
        ttk.Button(
            btn_frame,
            text="🌐 Open in Browser",
            command=self._open_in_browser
        ).pack(side="left", padx=(0, 8))
        ToolTip(btn_frame.winfo_children()[-1], "Open full-size interactive graph in web browser")
        
        ttk.Button(
            btn_frame,
            text="💾 Export Graph",
            command=self._export_graph
        ).pack(side="left", padx=(0, 8))
        ToolTip(btn_frame.winfo_children()[-1], "Export graph data as JSON")
        
        ttk.Button(
            btn_frame,
            text="📊 Export GraphML",
            command=self._export_graphml
        ).pack(side="left", padx=(0, 8))
        ToolTip(btn_frame.winfo_children()[-1], "Export for Gephi, Neo4j, or other graph tools")
        
        # Graph display area (placeholder until data loaded)
        self.graph_frame = ttk.LabelFrame(self, text="Entity Graph", padding=8, style="Panel.TLabelframe")
        self.graph_frame.pack(fill="both", expand=True, pady=(0, 8))
        
        self.graph_placeholder = ttk.Label(
            self.graph_frame,
            text="No graph data loaded.\n\nClick 'Refresh Graph' to extract entities from your transcripts,\nor process transcripts with GraphRAG enabled.",
            style="Muted.TLabel",
            justify="center"
        )
        self.graph_placeholder.pack(expand=True, pady=40)
        
        # Entity details panel (collapsible)
        self.details_frame = ttk.LabelFrame(self, text="Entity Details", padding=8, style="Panel.TLabelframe")
        self.details_frame.pack(fill="x", pady=(0, 8))
        
        self.details_text = tk.Text(
            self.details_frame, 
            height=6, 
            wrap=tk.WORD, 
            state="disabled",
            font=("JetBrains Mono", 9)
        )
        self.details_text.pack(fill="both", expand=True)
        
        # Store graph data
        self.entities: List[Dict] = []
        self.relationships: List[Dict] = []
        self.transcript_connections: Dict[str, List[str]] = {}  # entity_id -> [transcript_ids]
        
    def _on_filter_change(self):
        """Handle entity type filter changes."""
        self._refresh_graph()
        
    def _on_confidence_change(self, value):
        """Handle confidence threshold slider change."""
        self.confidence_label.configure(text=f"{float(value):.2f}")
        self._refresh_graph()
        
    def _on_layout_change(self, event=None):
        """Handle layout algorithm change."""
        self._refresh_graph()
        
    def set_graph_data(self, entities: List[Dict], relationships: List[Dict], 
                       transcript_connections: Optional[Dict] = None):
        """
        Load graph data from GraphRAG extraction.
        
        Args:
            entities: List of entity dicts with keys: name, type, description, confidence
            relationships: List of relationship dicts with keys: source, target, relationship, strength
            transcript_connections: Map of entity names to transcript IDs
        """
        self.entities = entities or []
        self.relationships = relationships or []
        self.transcript_connections = transcript_connections or {}
        
        # Update stats
        self.stats_labels['entities'].configure(text=f"Entities: {len(self.entities)}")
        self.stats_labels['relationships'].configure(text=f"Relationships: {len(self.relationships)}")
        
        if self.entities:
            avg_conf = sum(e.get('confidence', 0.5) for e in self.entities) / len(self.entities)
            self.stats_labels['confidence'].configure(text=f"Avg Confidence: {avg_conf:.2f}")
        
        # Count unique transcripts
        all_transcripts = set()
        for tids in self.transcript_connections.values():
            all_transcripts.update(tids)
        self.stats_labels['transcripts'].configure(text=f"Transcripts: {len(all_transcripts)}")
        
        self._refresh_graph()
        
    def _get_filtered_data(self):
        """Get entities and relationships filtered by current settings."""
        min_conf = self.confidence_var.get()
        active_types = [t for t, v in self.entity_filters.items() if v.get()]
        
        # Filter entities
        filtered_entities = [
            e for e in self.entities
            if e.get('confidence', 0.5) >= min_conf
            and e.get('type', 'concept').lower() in active_types
        ]
        
        # Get names of filtered entities for relationship filtering
        entity_names = {e['name'] for e in filtered_entities}
        
        # Filter relationships to only include edges between visible nodes
        filtered_rels = [
            r for r in self.relationships
            if r.get('source') in entity_names and r.get('target') in entity_names
        ]
        
        return filtered_entities, filtered_rels
        
    def _refresh_graph(self):
        """Regenerate and display the graph with current filters."""
        entities, relationships = self._get_filtered_data()
        
        if not entities:
            # Show placeholder
            self.graph_placeholder.pack(expand=True, pady=40)
            return
        
        # Hide placeholder
        self.graph_placeholder.pack_forget()
        
        # Generate vis.js HTML
        html = self._generate_vis_html(entities, relationships)
        
        # For tkinter, we can't embed vis.js directly, so we'll show a summary
        # and offer to open in browser
        summary = f"Graph loaded: {len(entities)} entities, {len(relationships)} relationships\n"
        summary += f"Click 'Open in Browser' for interactive visualization."
        
        self._set_details(summary)
        
        # Store HTML for browser view
        self._current_html = html
        
    def _generate_vis_html(self, entities: List[Dict], relationships: List[Dict]) -> str:
        """Generate vis.js HTML for the knowledge graph."""
        
        # Entity type colors
        type_colors = {
            'person': '#4ECDC4',
            'organization': '#FF6B6B',
            'location': '#45B7D1',
            'concept': '#F7DC6F',
            'event': '#BB8FCE',
            'product': '#85C1E2',
        }
        
        # Convert entities to vis.js nodes
        nodes = []
        for i, entity in enumerate(entities):
            etype = entity.get('type', 'concept').lower()
            confidence = entity.get('confidence', 0.5)
            
            node = {
                'id': i,
                'label': entity['name'] if self.show_labels_var.get() else '',
                'title': f"<b>{entity['name']}</b><br>Type: {etype}<br>Confidence: {confidence:.2f}<br><br>{entity.get('description', '')}",
                'color': type_colors.get(etype, '#888888'),
                'size': 10 + (confidence * 20),
                'group': etype,
                'font': {'size': 12, 'color': '#ffffff'},
            }
            nodes.append(node)
        
        # Create name-to-id mapping for edges
        name_to_id = {e['name']: i for i, e in enumerate(entities)}
        
        # Convert relationships to vis.js edges
        edges = []
        for rel in relationships:
            source_id = name_to_id.get(rel.get('source'))
            target_id = name_to_id.get(rel.get('target'))
            
            if source_id is not None and target_id is not None:
                edge = {
                    'from': source_id,
                    'to': target_id,
                    'label': rel.get('relationship', ''),
                    'title': rel.get('relationship', 'related'),
                    'width': 1 + (rel.get('strength', 0.5) * 3),
                    'color': {'color': '#888888', 'opacity': 0.6},
                    'font': {'size': 10, 'align': 'middle'},
                }
                edges.append(edge)
        
        # Get vis.js library path
        lib_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'lib', 'vis-9.1.2'
        )
        
        layout = self.layout_var.get()
        
        html = f'''<!DOCTYPE html>
<html>
<head>
    <title>PlaudBlender Knowledge Graph</title>
    <script type="text/javascript" src="file://{lib_path}/vis-network.min.js"></script>
    <link href="file://{lib_path}/vis-network.css" rel="stylesheet" type="text/css" />
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background: #1a1a2e;
            color: #ffffff;
        }}
        #header {{
            padding: 16px;
            background: #16213e;
            border-bottom: 1px solid #0f3460;
        }}
        #header h1 {{
            margin: 0;
            font-size: 24px;
            color: #e94560;
        }}
        #stats {{
            margin-top: 8px;
            font-size: 14px;
            color: #888888;
        }}
        #legend {{
            position: absolute;
            top: 80px;
            right: 16px;
            background: rgba(22, 33, 62, 0.9);
            padding: 12px;
            border-radius: 8px;
            z-index: 1000;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin-bottom: 6px;
            font-size: 12px;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        #graph {{
            width: 100%;
            height: calc(100vh - 80px);
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>🕸️ PlaudBlender Knowledge Graph</h1>
        <div id="stats">{len(entities)} entities • {len(edges)} relationships</div>
    </div>
    <div id="legend">
        <div class="legend-item"><div class="legend-color" style="background: #4ECDC4"></div>Person</div>
        <div class="legend-item"><div class="legend-color" style="background: #FF6B6B"></div>Organization</div>
        <div class="legend-item"><div class="legend-color" style="background: #45B7D1"></div>Location</div>
        <div class="legend-item"><div class="legend-color" style="background: #F7DC6F"></div>Concept</div>
        <div class="legend-item"><div class="legend-color" style="background: #BB8FCE"></div>Event</div>
        <div class="legend-item"><div class="legend-color" style="background: #85C1E2"></div>Product</div>
    </div>
    <div id="graph"></div>
    
    <script type="text/javascript">
        var nodes = new vis.DataSet({json.dumps(nodes)});
        var edges = new vis.DataSet({json.dumps(edges)});
        
        var container = document.getElementById('graph');
        var data = {{ nodes: nodes, edges: edges }};
        
        var options = {{
            nodes: {{
                shape: 'dot',
                borderWidth: 2,
                shadow: true,
            }},
            edges: {{
                smooth: {{
                    type: 'continuous'
                }},
                arrows: {{
                    to: {{ enabled: true, scaleFactor: 0.5 }}
                }}
            }},
            physics: {{
                solver: '{layout}',
                {layout}: {{
                    gravitationalConstant: -50,
                    springLength: 100,
                    springConstant: 0.05
                }},
                stabilization: {{
                    iterations: 150
                }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 100,
                navigationButtons: true,
                keyboard: true
            }}
        }};
        
        var network = new vis.Network(container, data, options);
        
        // Click handler for entity details
        network.on("click", function(params) {{
            if (params.nodes.length > 0) {{
                var nodeId = params.nodes[0];
                var node = nodes.get(nodeId);
                console.log("Selected:", node);
            }}
        }});
    </script>
</body>
</html>'''
        
        return html
        
    def _open_in_browser(self):
        """Open the current graph in the system's default web browser."""
        if not hasattr(self, '_current_html') or not self._current_html:
            entities, relationships = self._get_filtered_data()
            if not entities:
                from tkinter import messagebox
                messagebox.showinfo("No Data", "No graph data to display. Extract entities first.")
                return
            self._current_html = self._generate_vis_html(entities, relationships)
        
        # Write to temp file and open
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(self._current_html)
            temp_path = f.name
        
        webbrowser.open(f'file://{temp_path}')
        
    def _export_graph(self):
        """Export graph data as JSON."""
        from tkinter import filedialog
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Export Graph Data"
        )
        
        if filepath:
            data = {
                'entities': self.entities,
                'relationships': self.relationships,
                'transcript_connections': self.transcript_connections,
                'exported_at': str(__import__('datetime').datetime.now())
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            from tkinter import messagebox
            messagebox.showinfo("Export Complete", f"Graph exported to:\n{filepath}")
            
    def _export_graphml(self):
        """Export as GraphML for Gephi/Neo4j."""
        from tkinter import filedialog
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".graphml",
            filetypes=[("GraphML files", "*.graphml"), ("All files", "*.*")],
            title="Export GraphML"
        )
        
        if filepath:
            # Generate GraphML XML
            xml = self._generate_graphml()
            
            with open(filepath, 'w') as f:
                f.write(xml)
            
            from tkinter import messagebox
            messagebox.showinfo("Export Complete", f"GraphML exported to:\n{filepath}")
            
    def _generate_graphml(self) -> str:
        """Generate GraphML XML format."""
        entities, relationships = self._get_filtered_data()
        
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
            '  <key id="label" for="node" attr.name="label" attr.type="string"/>',
            '  <key id="type" for="node" attr.name="type" attr.type="string"/>',
            '  <key id="confidence" for="node" attr.name="confidence" attr.type="double"/>',
            '  <key id="description" for="node" attr.name="description" attr.type="string"/>',
            '  <key id="relationship" for="edge" attr.name="relationship" attr.type="string"/>',
            '  <key id="strength" for="edge" attr.name="strength" attr.type="double"/>',
            '  <graph id="G" edgedefault="directed">',
        ]
        
        # Add nodes
        for i, entity in enumerate(entities):
            lines.append(f'    <node id="n{i}">')
            lines.append(f'      <data key="label">{self._xml_escape(entity["name"])}</data>')
            lines.append(f'      <data key="type">{entity.get("type", "concept")}</data>')
            lines.append(f'      <data key="confidence">{entity.get("confidence", 0.5)}</data>')
            lines.append(f'      <data key="description">{self._xml_escape(entity.get("description", ""))}</data>')
            lines.append('    </node>')
        
        # Create name-to-id mapping
        name_to_id = {e['name']: i for i, e in enumerate(entities)}
        
        # Add edges
        edge_id = 0
        for rel in relationships:
            source_id = name_to_id.get(rel.get('source'))
            target_id = name_to_id.get(rel.get('target'))
            
            if source_id is not None and target_id is not None:
                lines.append(f'    <edge id="e{edge_id}" source="n{source_id}" target="n{target_id}">')
                lines.append(f'      <data key="relationship">{self._xml_escape(rel.get("relationship", "related"))}</data>')
                lines.append(f'      <data key="strength">{rel.get("strength", 0.5)}</data>')
                lines.append('    </edge>')
                edge_id += 1
        
        lines.append('  </graph>')
        lines.append('</graphml>')
        
        return '\n'.join(lines)
        
    def _xml_escape(self, text: str) -> str:
        """Escape XML special characters."""
        return (str(text)
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&apos;'))
        
    def _set_details(self, text: str):
        """Update the details text area."""
        self.details_text.configure(state="normal")
        self.details_text.delete('1.0', tk.END)
        self.details_text.insert('1.0', text)
        self.details_text.configure(state="disabled")
