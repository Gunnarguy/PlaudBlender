"""
Generate beautiful interactive mind map from all processed transcripts
"""
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
import logging

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.notion_client import NotionTranscriptClient
from src.visualizer import MindMapGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


def create_mind_map(output_file="output/knowledge_graph.html"):
    """
    Generate interactive mind map from all processed transcripts
    
    Args:
        output_file: Path for output HTML file
    """
    print("\n" + "=" * 60)
    print("🎨 GENERATING MIND MAP")
    print("=" * 60 + "\n")
    
    start_time = datetime.now()
    
    try:
        # Initialize clients
        logger.info("Initializing Notion client...")
        notion = NotionTranscriptClient()
        
        logger.info("Initializing visualizer...")
        visualizer = MindMapGenerator()
        
        # Get all processed transcripts
        logger.info("Fetching all processed transcripts from Notion...")
        processed_pages = notion.fetch_all_processed_transcripts()
        
        if not processed_pages:
            print("\n❌ No processed transcripts found!")
            print("💡 Tip: First get transcripts into Notion (via the GUI Notion sync, or your ingestion pipeline), then re-run this script.\n")
            return None
        
        print(f"\n📊 Found {len(processed_pages)} processed transcripts")
        
        # Build graph
        print("\n🔨 Building knowledge graph...\n")
        
        for idx, page in enumerate(processed_pages, 1):
            page_id = page["id"]
            
            # Get title
            title = notion.get_page_title(page)
            
            # Extract themes from page properties
            themes_prop = page["properties"].get("Themes", {}).get("rich_text", [])
            if themes_prop and themes_prop[0].get("plain_text"):
                themes = [t.strip() for t in themes_prop[0]["plain_text"].split(",")]
            else:
                themes = ["General"]
            
            # Get creation date
            created_prop = page.get("created_time", "")
            
            # Count connections
            connections_prop = page["properties"].get("Connections", {}).get("rich_text", [])
            connection_count = 0
            connection_ids = []
            
            if connections_prop and connections_prop[0].get("plain_text"):
                conn_text = connections_prop[0]["plain_text"]
                connection_ids = [c.strip() for c in conn_text.split(",") if c.strip()]
                connection_count = len(connection_ids)
            
            # Add node with centrality based on connections
            centrality = min(connection_count / 5.0, 1.0)  # Normalize
            
            visualizer.add_transcript_node(
                page_id=page_id,
                title=title,
                themes=themes,
                centrality=centrality,
                metadata={
                    'created': created_prop,
                    'connection_count': connection_count
                }
            )
            
            # Add edges for connections
            for conn_id in connection_ids:
                visualizer.add_connection(page_id, conn_id, "relates_to", 1.0)
            
            print(f"  [{idx}/{len(processed_pages)}] Added: {title[:50]}")
        
        # Generate statistics
        print("\n📈 Generating statistics...")
        stats = visualizer.generate_statistics()
        
        print(f"\n📊 Network Statistics:")
        print(f"  • Total Nodes: {stats['total_nodes']}")
        print(f"  • Total Connections: {stats['total_edges']}")
        print(f"  • Average Connections: {stats['avg_connections']:.1f}")
        print(f"  • Network Density: {stats['network_density']:.2%}")
        print(f"  • Isolated Nodes: {stats['isolated_nodes']}")
        
        if stats['themes']:
            print(f"\n🏷️  Top Themes:")
            for theme, count in list(stats['themes'].items())[:5]:
                print(f"  • {theme}: {count}")
        
        if stats['most_connected']:
            print(f"\n⭐ Most Connected Transcripts:")
            for node in stats['most_connected'][:3]:
                print(f"  • {node['title'][:40]}: {node['connections']} links")
        
        # Generate interactive visualization
        print(f"\n✨ Generating interactive HTML...")
        output_path = visualizer.generate_interactive_map(output_file)
        
        # Export JSON data
        json_path = visualizer.export_graph_data("output/graph_data.json")
        
        duration = (datetime.now() - start_time).total_seconds()
        
        print("\n" + "=" * 60)
        print("✅ MIND MAP GENERATED SUCCESSFULLY!")
        print("=" * 60)
        print(f"\n📁 Files created:")
        print(f"  • HTML: {os.path.abspath(output_path)}")
        if json_path:
            print(f"  • JSON: {os.path.abspath(json_path)}")
        print(f"\n⏱️  Generation time: {duration:.2f} seconds")
        print(f"\n🌐 Open in browser: file://{os.path.abspath(output_path)}")
        print()
        
        return output_path
        
    except Exception as e:
        print(f"\n❌ Error generating mind map: {str(e)}")
        logger.exception("Full error details:")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate interactive mind map from transcripts")
    parser.add_argument(
        '-o', '--output',
        default='output/knowledge_graph.html',
        help='Output HTML file path (default: output/knowledge_graph.html)'
    )
    
    args = parser.parse_args()
    
    result = create_mind_map(args.output)
    
    if result:
        sys.exit(0)
    else:
        sys.exit(1)
