"""Legacy Pinecone view stub.

Even though the project is migrating to Qdrant-first, some UI smoke tests still
expect this module to exist.
"""

from __future__ import annotations

from dataclasses import dataclass

from gui.views.base import BaseView


@dataclass
class PineconeView(BaseView):
    name: str = "pinecone"
