"""Legacy Pinecone service stub.

The codebase is transitioning to Qdrant-first, but smoke tests import this.
"""

from __future__ import annotations


class PineconeService:
    def __init__(self):
        self.enabled = False
