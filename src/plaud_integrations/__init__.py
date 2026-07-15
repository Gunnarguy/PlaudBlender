"""Public PLAUD integration adapters kept separate by authentication model."""

from .account_protocol import PlaudAccountSource
from .embedded_auth import PlaudEmbeddedAuthClient, PlaudRegion
from .embedded_upload import PlaudEmbeddedUploadClient
from .legacy_account import PlaudLegacyAccountAdapter
from .mcp_account import PlaudMCPAccountAdapter
from .transcription import PlaudTranscriptionClient

__all__ = [
    "PlaudAccountSource",
    "PlaudEmbeddedAuthClient",
    "PlaudEmbeddedUploadClient",
    "PlaudLegacyAccountAdapter",
    "PlaudMCPAccountAdapter",
    "PlaudRegion",
    "PlaudTranscriptionClient",
]
