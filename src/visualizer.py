"""
Beautiful mind map visualization with interactive dashboard
"""
from pyvis.network import Network
import networkx as nx
from datetime import datetime
import json
import logging
from collections import Counter
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MindMapGenerator:
    """Generate beautiful interactive knowledge graphs"""
    
    def __init__(self):
        """Initialize graph"""
        self.graph = nx.Graph()
        self.node_metadata = {}  # Store additional node information
        self.theme_colors = {}  # Map themes to colors
        self.color_palette = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8",
            "#F7DC6F", "#BB8FCE", "#85C1E2", "#F8B739", "#52B788",
            "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"
        ]
    
    def _get_theme_color(self, theme):
        """Get consistent color for a theme"""
        if theme not in self.theme_colors:
            color_idx = len(self.theme_colors) % len(self.color_palette)
            self.theme_colors[theme] = self.color_palette[color_idx]
        return self.theme_colors[theme]
    
    def add_transcript_node(self, page_id, title, themes, centrality=1.0, metadata=None):
        """
        Add transcript as node with themes
        
        Args:
            page_id: Unique page identifier
            title: Node title/label
            themes: List of theme strings
            centrality: Importance score (affects size)
            metadata: Additional node metadata
        """
        # Determine primary theme
        primary_theme = themes[0] if themes else "General"
        color = self._get_theme_color(primary_theme)
        
        # Calculate node size based on centrality
        size = 20 + (centrality * 30)
        
        # Create hover tooltip
        tooltip = f"<b>{title}</b><br><br>"
        tooltip += f"<b>Themes:</b> {', '.join(themes)}<br>"
        if metadata:
            if metadata.get('created'):
                tooltip += f"<b>Created:</b> {metadata['created']}<br>"
            if metadata.get('word_count'):
                tooltip += f"<b>Words:</b> {metadata['word_count']}<br>"
        
        # Add node to graph
        self.graph.add_node(
            page_id,
            label=title[:40] + ("..." if len(title) > 40 else ""),
            title=tooltip,
            size=size,
            color=color,
            group=primary_theme,
            themes=themes
        )
        
        # Store metadata
        self.node_metadata[page_id] = {
            "title": title,
            "themes": themes,
            "centrality": centrality,
            "metadata": metadata or {}
        }
        
        logger.info(f"Added node: {title[:30]} (theme: {primary_theme})")
    
    def add_connection(self, source_id, target_id, relationship="relates_to", weight=1.0):
        """
        Add edge between transcripts
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relationship: Edge label
            weight: Connection strength
        """
        if source_id in self.graph.nodes and target_id in self.graph.nodes:
            self.graph.add_edge(
                source_id,
                target_id,
                title=relationship,
                weight=weight,
                color={'color': '#888888', 'opacity': 0.5}
            )
            logger.debug(f"Connected: {source_id[:8]}... -> {target_id[:8]}...")
    
    def calculate_centralities(self):
        """Calculate node importance using various centrality measures"""
        if len(self.graph.nodes) == 0:
            return {}
        
        try:
            # Degree centrality (number of connections)
            degree_cent = nx.degree_centrality(self.graph)
            
            # Betweenness centrality (bridge nodes)
            if len(self.graph.nodes) > 2:
                between_cent = nx.betweenness_centrality(self.graph)
            else:
                between_cent = {node: 0 for node in self.graph.nodes}
            
            # Combine metrics
            centralities = {}
            for node in self.graph.nodes:
                centralities[node] = (degree_cent.get(node, 0) * 0.6 + 
                                     between_cent.get(node, 0) * 0.4)
            
            return centralities
            
        except Exception as e:
            logger.error(f"Error calculating centralities: {e}")
            return {node: 0.5 for node in self.graph.nodes}
    
    def generate_statistics(self):
        """Generate network statistics for dashboard"""
        stats = {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "themes": {},
            "avg_connections": 0,
            "most_connected": [],
            "isolated_nodes": 0,
            "network_density": 0
        }
        
        if stats["total_nodes"] == 0:
            return stats
        
        # Theme distribution
        all_themes = []
        for node in self.graph.nodes:
            node_data = self.node_metadata.get(node, {})
            all_themes.extend(node_data.get("themes", ["General"]))
        
        theme_counts = Counter(all_themes)
        stats["themes"] = dict(theme_counts.most_common(10))
        
        # Connection statistics
        degrees = dict(self.graph.degree())
        if degrees:
            stats["avg_connections"] = sum(degrees.values()) / len(degrees)
            
            # Most connected nodes
            sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
            stats["most_connected"] = [
                {
                    "id": node,
                    "title": self.node_metadata.get(node, {}).get("title", "Unknown"),
                    "connections": degree
                }
                for node, degree in sorted_nodes[:5]
            ]
            
            # Isolated nodes
            stats["isolated_nodes"] = sum(1 for d in degrees.values() if d == 0)
        
        # Network density
        if stats["total_nodes"] > 1:
            stats["network_density"] = nx.density(self.graph)
        
        return stats
    
    def generate_interactive_map(self, output_file="output/knowledge_graph.html"):
        """
        Create interactive HTML visualization with enhanced styling
        
        Args:
            output_file: Output HTML file path
            
        Returns:
            Path to generated file
        """
        logger.info("Generating interactive mind map...")
        
        # Calculate centralities to adjust node sizes
        centralities = self.calculate_centralities()
        for node in self.graph.nodes:
            current_size = self.graph.nodes[node].get('size', 20)
            centrality_boost = centralities.get(node, 0) * 20
            self.graph.nodes[node]['size'] = current_size + centrality_boost
        
        # Create Pyvis network
        net = Network(
            height='900px',
            width='100%',
            bgcolor='#0a0a0a',
            font_color='#ffffff',
            notebook=False,
            heading="🧠 Knowledge Mind Map"
        )
        
        # Import NetworkX graph
        net.from_nx(self.graph)
        
        # Enhanced physics settings for organic layout
        net.set_options("""
        {
          "nodes": {
            "borderWidth": 2,
            "borderWidthSelected": 4,
            "font": {
              "size": 14,
              "color": "#ffffff",
              "face": "arial",
              "strokeWidth": 2,
              "strokeColor": "#000000"
            },
            "shadow": {
              "enabled": true,
              "color": "rgba(0,0,0,0.5)",
              "size": 10,
              "x": 3,
              "y": 3
            }
          },
          "edges": {
            "color": {
              "inherit": false,
              "opacity": 0.4
            },
            "smooth": {
              "enabled": true,
              "type": "continuous",
              "roundness": 0.5
            },
            "width": 2,
            "shadow": {
              "enabled": true
            }
          },
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -80,
              "centralGravity": 0.015,
              "springLength": 200,
              "springConstant": 0.08,
              "damping": 0.4,
              "avoidOverlap": 0.5
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {
              "enabled": true,
              "iterations": 200,
              "updateInterval": 25
            }
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "hideEdgesOnDrag": true,
            "navigationButtons": true,
            "keyboard": {
              "enabled": true
            }
          }
        }
        """)
        
        # Generate HTML file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        net.save_graph(output_file)
        
        # Add custom CSS and statistics
        self._enhance_html(output_file)
        
        logger.info(f"✨ Mind map generated: {output_file}")
        return output_file
    
    def _enhance_html(self, html_file):
        """Add custom CSS and statistics panel to HTML"""
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Generate statistics
            stats = self.generate_statistics()
            
            # Create statistics HTML
            stats_html = f"""
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 0;
                }}
                #stats-panel {{
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: rgba(15, 15, 15, 0.95);
                    border: 2px solid #4ECDC4;
                    border-radius: 12px;
                    padding: 20px;
                    color: white;
                    width: 280px;
                    max-height: 80vh;
                    overflow-y: auto;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
                    backdrop-filter: blur(10px);
                    z-index: 1000;
                }}
                #stats-panel h2 {{
                    margin-top: 0;
                    color: #4ECDC4;
                    font-size: 20px;
                    border-bottom: 2px solid #4ECDC4;
                    padding-bottom: 10px;
                }}
                .stat-item {{
                    margin: 12px 0;
                    padding: 8px;
                    background: rgba(255,255,255,0.05);
                    border-radius: 6px;
                }}
                .stat-label {{
                    font-weight: bold;
                    color: #4ECDC4;
                    font-size: 12px;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }}
                .stat-value {{
                    font-size: 24px;
                    margin-top: 5px;
                    color: #ffffff;
                }}
                .theme-item {{
                    margin: 6px 0;
                    padding: 6px 10px;
                    background: rgba(78, 205, 196, 0.2);
                    border-left: 3px solid #4ECDC4;
                    border-radius: 4px;
                    font-size: 13px;
                }}
                .connected-node {{
                    margin: 6px 0;
                    padding: 6px 10px;
                    background: rgba(255, 107, 107, 0.2);
                    border-left: 3px solid #FF6B6B;
                    border-radius: 4px;
                    font-size: 12px;
                }}
                #toggle-stats {{
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: #4ECDC4;
                    color: #0a0a0a;
                    border: none;
                    padding: 12px 20px;
                    border-radius: 8px;
                    cursor: pointer;
                    font-weight: bold;
                    z-index: 1001;
                    font-size: 14px;
                    box-shadow: 0 4px 12px rgba(78, 205, 196, 0.4);
                }}
                #toggle-stats:hover {{
                    background: #45B7D1;
                    transform: scale(1.05);
                }}
                .legend {{
                    margin-top: 20px;
                    padding-top: 15px;
                    border-top: 1px solid rgba(255,255,255,0.2);
                }}
                .legend-title {{
                    font-weight: bold;
                    color: #4ECDC4;
                    margin-bottom: 10px;
                }}
            </style>
            <button id="toggle-stats" onclick="toggleStats()">📊 Stats</button>
            <div id="stats-panel" style="display:none;">
                <h2>📊 Network Statistics</h2>
                
                <div class="stat-item">
                    <div class="stat-label">Total Transcripts</div>
                    <div class="stat-value">{stats['total_nodes']}</div>
                </div>
                
                <div class="stat-item">
                    <div class="stat-label">Connections</div>
                    <div class="stat-value">{stats['total_edges']}</div>
                </div>
                
                <div class="stat-item">
                    <div class="stat-label">Avg Connections</div>
                    <div class="stat-value">{stats['avg_connections']:.1f}</div>
                </div>
                
                <div class="stat-item">
                    <div class="stat-label">Network Density</div>
                    <div class="stat-value">{stats['network_density']:.2%}</div>
                </div>
                
                <div class="legend">
                    <div class="legend-title">🏷️ Top Themes</div>
                    {"".join(f'<div class="theme-item">{theme}: {count}</div>' 
                             for theme, count in list(stats['themes'].items())[:5])}
                </div>
                
                <div class="legend">
                    <div class="legend-title">⭐ Most Connected</div>
                    {"".join(f'<div class="connected-node">{node["title"][:30]}: {node["connections"]} links</div>' 
                             for node in stats['most_connected'][:5])}
                </div>
            </div>
            <script>
                function toggleStats() {{
                    var panel = document.getElementById('stats-panel');
                    var button = document.getElementById('toggle-stats');
                    if (panel.style.display === 'none') {{
                        panel.style.display = 'block';
                        button.textContent = '✖️ Close';
                    }} else {{
                        panel.style.display = 'none';
                        button.textContent = '📊 Stats';
                    }}
                }}
            </script>
            """
            
            # Insert before closing body tag
            html_content = html_content.replace('</body>', f'{stats_html}</body>')
            
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info("Enhanced HTML with statistics panel")
            
        except Exception as e:
            logger.error(f"Error enhancing HTML: {e}")
    
    def export_graph_data(self, output_file="output/graph_data.json"):
        """Export graph data as JSON for further analysis"""
        try:
            data = {
                "nodes": [
                    {
                        "id": node,
                        "data": self.node_metadata.get(node, {})
                    }
                    for node in self.graph.nodes
                ],
                "edges": [
                    {
                        "source": edge[0],
                        "target": edge[1],
                        "data": self.graph.edges[edge]
                    }
                    for edge in self.graph.edges
                ],
                "statistics": self.generate_statistics(),
                "generated_at": datetime.now().isoformat()
            }
            
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported graph data to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error exporting graph data: {e}")
            return None
