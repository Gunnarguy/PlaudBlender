"""
Notion Client for fetching and updating transcript pages
"""
import os
from notion_client import Client
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class NotionTranscriptClient:
    """Client for interacting with Notion transcript database"""
    
    def __init__(self):
        """Initialize Notion client"""
        self.client = Client(auth=os.getenv("NOTION_TOKEN"))
        self.database_id = os.getenv("NOTION_DATABASE_ID")
        
        if not self.client or not self.database_id:
            raise ValueError("NOTION_TOKEN and NOTION_DATABASE_ID must be set in .env file")
        
        logger.info(f"Notion client initialized for database: {self.database_id[:8]}...")
    
    def fetch_new_transcripts(self, minutes_ago=20):
        """
        Fetch transcripts created in last N minutes with Status='New'
        
        Args:
            minutes_ago: Number of minutes to look back
            
        Returns:
            List of Notion page objects
        """
        last_check = (datetime.now() - timedelta(minutes=minutes_ago)).isoformat()
        
        try:
            response = self.client.request(
                path=f"databases/{self.database_id}/query",
                method="POST",
                body={
                    "filter": {
                        "and": [
                            {"property": "Status", "select": {"equals": "New"}},
                            {"property": "Created", "date": {"after": last_check}}
                        ]
                    },
                    "page_size": 100  # Increased for batch processing
                }
            )
            
            results = response.get("results", [])
            logger.info(f"Found {len(results)} new transcripts")
            return results
            
        except Exception as e:
            logger.error(f"Error fetching transcripts: {e}")
            return []
    
    def fetch_all_processed_transcripts(self):
        """
        Fetch all processed transcripts for mind map generation
        
        Returns:
            List of all processed Notion pages
        """
        try:
            all_results = []
            has_more = True
            start_cursor = None
            
            while has_more:
                body = {
                    "filter": {"property": "Status", "select": {"equals": "Processed"}},
                    "page_size": 100
                }
                
                if start_cursor:
                    body["start_cursor"] = start_cursor
                
                response = self.client.request(
                    path=f"databases/{self.database_id}/query",
                    method="POST",
                    body=body
                )
                all_results.extend(response.get("results", []))
                
                has_more = response.get("has_more", False)
                start_cursor = response.get("next_cursor")
            
            logger.info(f"Retrieved {len(all_results)} processed transcripts")
            return all_results
            
        except Exception as e:
            logger.error(f"Error fetching processed transcripts: {e}")
            return []
    
    def get_page_content(self, page_id):
        """
        Extract text content from Notion page
        
        Args:
            page_id: Notion page ID
            
        Returns:
            Combined text content from all blocks
        """
        try:
            blocks = self.client.blocks.children.list(page_id)
            text_parts = []
            
            for block in blocks.get("results", []):
                block_type = block.get("type")
                
                if block_type == "paragraph" and block.get("paragraph", {}).get("rich_text"):
                    for rich_text in block["paragraph"]["rich_text"]:
                        text_parts.append(rich_text.get("plain_text", ""))
                
                elif block_type == "heading_1" and block.get("heading_1", {}).get("rich_text"):
                    for rich_text in block["heading_1"]["rich_text"]:
                        text_parts.append(rich_text.get("plain_text", ""))
                
                elif block_type == "heading_2" and block.get("heading_2", {}).get("rich_text"):
                    for rich_text in block["heading_2"]["rich_text"]:
                        text_parts.append(rich_text.get("plain_text", ""))
                
                elif block_type == "heading_3" and block.get("heading_3", {}).get("rich_text"):
                    for rich_text in block["heading_3"]["rich_text"]:
                        text_parts.append(rich_text.get("plain_text", ""))
                
                elif block_type == "bulleted_list_item" and block.get("bulleted_list_item", {}).get("rich_text"):
                    for rich_text in block["bulleted_list_item"]["rich_text"]:
                        text_parts.append("• " + rich_text.get("plain_text", ""))
            
            text = "\n".join(text_parts).strip()
            logger.info(f"Extracted {len(text)} characters from page {page_id[:8]}...")
            return text
            
        except Exception as e:
            logger.error(f"Error getting page content for {page_id}: {e}")
            return ""
    
    def get_page_title(self, page):
        """
        Extract title from Notion page object
        
        Args:
            page: Notion page object
            
        Returns:
            Page title as string
        """
        try:
            title_prop = page.get("properties", {}).get("Title", {})
            if title_prop.get("title"):
                return title_prop["title"][0].get("plain_text", "Untitled")
            return "Untitled"
        except Exception:
            return "Untitled"
    
    def update_page_with_synthesis(self, page_id, synthesis, connections, themes=None):
        """
        Update Notion page with processing results
        
        Args:
            page_id: Notion page ID
            synthesis: LLM-generated synthesis text
            connections: List of related page IDs
            themes: List of extracted themes
        """
        try:
            properties = {
                "Status": {"select": {"name": "Processed"}},
                "ProcessedAt": {"date": {"start": datetime.now().isoformat()}}
            }
            
            # Add synthesis (truncate to fit Notion's limits)
            if synthesis:
                properties["Synthesis"] = {
                    "rich_text": [{"text": {"content": synthesis[:2000]}}]
                }
            
            # Add connections as comma-separated list
            if connections:
                conn_text = ", ".join([c[:8] for c in connections])  # Shortened IDs
                properties["Connections"] = {
                    "rich_text": [{"text": {"content": conn_text[:2000]}}]
                }
            
            # Add themes if property exists
            if themes:
                properties["Themes"] = {
                    "rich_text": [{"text": {"content": ", ".join(themes[:10])}}]
                }
            
            self.client.pages.update(page_id=page_id, properties=properties)
            logger.info(f"Updated page {page_id[:8]}... with synthesis and {len(connections)} connections")
            
        except Exception as e:
            logger.error(f"Error updating page {page_id}: {e}")
    
    def mark_page_as_processing(self, page_id):
        """Mark a page as currently being processed"""
        try:
            self.client.pages.update(
                page_id=page_id,
                properties={"Status": {"select": {"name": "Processing"}}}
            )
            logger.info(f"Marked page {page_id[:8]}... as Processing")
        except Exception as e:
            logger.error(f"Error marking page as processing: {e}")
