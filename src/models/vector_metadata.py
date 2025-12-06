"""
Vector metadata schema for PlaudBlender.

Defines required and optional fields for all vectors stored in Pinecone.
Use validate_metadata() before any upsert to ensure consistency.
"""
import hashlib
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class VectorMetadata(BaseModel):
    """
    Canonical metadata schema for Pinecone vectors.

    Required fields ensure we can always identify, filter, and dedupe vectors.
    Optional fields enrich search and display.
    """

    # ── Required ──────────────────────────────────────────────────────────────
    recording_id: str = Field(..., description="Unique ID of the source recording")
    source: str = Field(default="plaud", description="Data source (plaud, manual, etc.)")
    model: str = Field(..., description="Embedding model used (e.g., gemini-embedding-001)")
    dimension: int = Field(..., ge=128, le=4096, description="Embedding dimension")
    text_hash: str = Field(..., description="SHA-256 hash of embedded text for dedup")

    # ── Recommended ───────────────────────────────────────────────────────────
    segment_id: Optional[str] = Field(default=None, description="Segment/chunk ID within recording")
    title: Optional[str] = Field(default=None, description="Display title")
    start_at: Optional[str] = Field(default=None, description="ISO timestamp of recording start")
    duration_ms: Optional[int] = Field(default=None, ge=0, description="Duration in milliseconds")
    themes: Optional[List[str]] = Field(default=None, description="Extracted themes/tags")
    provider: Optional[str] = Field(default=None, description="Embedding provider (google, openai, pinecone)")

    # ── Internal ──────────────────────────────────────────────────────────────
    version: int = Field(default=1, description="Schema version for future migrations")
    indexed_at: Optional[str] = Field(default=None, description="ISO timestamp when indexed")

    @field_validator("text_hash")
    @classmethod
    def validate_hash(cls, v: str) -> str:
        if len(v) != 64:
            raise ValueError("text_hash must be a 64-char SHA-256 hex string")
        return v

    def to_pinecone(self) -> Dict[str, Any]:
        """Convert to Pinecone-compatible metadata dict (no None values)."""
        data = self.model_dump(exclude_none=True)
        # Pinecone stores lists as JSON; flatten themes to comma-separated if needed
        if "themes" in data and isinstance(data["themes"], list):
            data["themes"] = ",".join(data["themes"])
        return data


def compute_text_hash(text: str) -> str:
    """Compute SHA-256 hash for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_metadata(meta: Dict[str, Any]) -> VectorMetadata:
    """
    Validate a metadata dict against the schema.

    Raises:
        pydantic.ValidationError: If required fields are missing or invalid
    """
    return VectorMetadata(**meta)


def build_metadata(
    recording_id: str,
    text: str,
    model: str,
    dimension: int,
    source: str = "plaud",
    provider: Optional[str] = None,
    title: Optional[str] = None,
    start_at: Optional[str] = None,
    duration_ms: Optional[int] = None,
    themes: Optional[List[str]] = None,
    segment_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a validated metadata dict for upsert.

    Args:
        recording_id: Unique recording identifier
        text: The text being embedded (used for hash)
        model: Embedding model name
        dimension: Embedding dimension
        ... (other optional fields)

    Returns:
        Pinecone-ready metadata dict
    """
    meta = VectorMetadata(
        recording_id=recording_id,
        source=source,
        model=model,
        dimension=dimension,
        text_hash=compute_text_hash(text),
        provider=provider,
        title=title,
        start_at=start_at,
        duration_ms=duration_ms,
        themes=themes,
        segment_id=segment_id,
        indexed_at=datetime.utcnow().isoformat() + "Z",
    )
    return meta.to_pinecone()
