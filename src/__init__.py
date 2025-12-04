"""
PlaudBlender - Voice Transcript Knowledge Graph

Transform your Plaud recordings into a searchable, visual knowledge graph.
"""

__version__ = "2.0.0"

# Only import core Plaud clients by default
# Other modules have heavy dependencies (pinecone, google-ai, etc.)
from .plaud_oauth import PlaudOAuthClient
from .plaud_client import PlaudClient

__all__ = [
    "PlaudOAuthClient",
    "PlaudClient", 
]
