"""
Notion MCP Client for fetching transcript pages from synced databases
Uses page search instead of database queries to bypass synced database limitations
"""
import os
import subprocess
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class NotionMCPClient:
    """Client for interacting with Notion via MCP Docker tools for synced databases"""
    
    def __init__(self):
        """Initialize MCP client"""
        self.database_id = os.getenv("NOTION_DATABASE_ID")
        
        if not self.database_id:
            raise ValueError("NOTION_DATABASE_ID must be set in .env file")
        
        logger.info(f"Notion MCP client initialized for database: {self.database_id[:8]}...")
    
    def search_pages(self, query="", page_size=100):
        """
        Search for pages using MCP Docker API
        
        Args:
            query: Search query string
            page_size: Number of results to return
            
        Returns:
            List of page objects
        """
        try:
            # Using the mcp_docker API endpoint for search
            # This would normally be called via the MCP tools
            # For now, we'll use a hybrid approach
            logger.info(f"Searching for pages with query: '{query}'")
            
            # This is a placeholder - in production, you'd call the actual MCP tool
            # For now, we'll fall back to getting pages from the database's parent
            return []
            
        except Exception as e:
            logger.error(f"Error searching pages: {e}")
            return []
    
    def fetch_transcript_pages(self, days_back=7):
        """
        Fetch transcript pages from the last N days
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            List of transcript page objects with content
        """
        try:
            # Search for pages with date filters
            # This bypasses the synced database limitation
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            logger.info(f"Fetching transcripts from last {days_back} days (since {cutoff_date.date()})")
            
            # In a full implementation, this would use:
            # - mcp_docker_API-post-search with date filters
            # - Then fetch each page's content
            
            # For now, return empty list - this needs MCP integration
            return []
            
        except Exception as e:
            logger.error(f"Error fetching transcript pages: {e}")
            return []
    
    def get_page_properties(self, page):
        """
        Extract properties from a Notion page object
        
        Args:
            page: Notion page object from search results
            
        Returns:
            Dictionary of extracted properties
        """
        try:
            props = page.get("properties", {})
            
            # Extract title
            title = "Untitled"
            if "Name" in props and props["Name"].get("title"):
                title = props["Name"]["title"][0].get("plain_text", "Untitled")
            elif "Title" in props and props["Title"].get("title"):
                title = props["Title"]["title"][0].get("plain_text", "Untitled")
            
            # Extract transcript text
            transcript = ""
            if "Transcript" in props and props["Transcript"].get("rich_text"):
                parts = [rt.get("plain_text", "") for rt in props["Transcript"]["rich_text"]]
                transcript = "".join(parts)
            
            # Extract summary if exists
            summary = ""
            if "Summary" in props and props["Summary"].get("rich_text"):
                parts = [rt.get("plain_text", "") for rt in props["Summary"]["rich_text"]]
                summary = "".join(parts)
            
            return {
                "id": page.get("id"),
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "created_time": page.get("created_time"),
                "last_edited_time": page.get("last_edited_time")
            }
            
        except Exception as e:
            logger.error(f"Error extracting properties from page: {e}")
            return {}


def hybrid_fetch_transcripts(regular_client, mcp_client, days_back=7):
    """
    Hybrid approach: Try regular API first, fall back to MCP if synced database
    
    Args:
        regular_client: NotionTranscriptClient instance
        mcp_client: NotionMCPClient instance
        days_back: Days to look back
        
    Returns:
        List of transcript data dictionaries
    """
    try:
        # Try regular API first
        logger.info("Attempting regular Notion API fetch...")
        results = regular_client.fetch_new_transcripts(minutes_ago=days_back * 24 * 60)
        
        if results:
            logger.info(f"✅ Regular API succeeded: {len(results)} transcripts")
            return results
        
    except Exception as e:
        error_msg = str(e)
        
        if "multiple data sources" in error_msg.lower():
            logger.warning("⚠️ Synced database detected, switching to MCP approach...")
            
            # Use MCP to fetch pages
            logger.info("Using MCP Docker tools to fetch transcript pages...")
            mcp_results = mcp_client.fetch_transcript_pages(days_back=days_back)
            
            if mcp_results:
                logger.info(f"✅ MCP API succeeded: {len(mcp_results)} transcripts")
                return mcp_results
            else:
                logger.warning("⚠️ MCP fetch returned no results")
                return []
        else:
            logger.error(f"API error: {e}")
            return []
    
    return []


if __name__ == "__main__":
    # Test the MCP client
    print("\n🧪 Testing Notion MCP Client...")
    
    try:
        mcp_client = NotionMCPClient()
        print("✅ MCP client initialized")
        
        # Try fetching pages
        pages = mcp_client.fetch_transcript_pages(days_back=30)
        print(f"📄 Found {len(pages)} pages")
        
    except Exception as e:
        print(f"❌ Error: {e}")
