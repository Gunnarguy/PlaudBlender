from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional
import os
import logging

from dotenv import load_dotenv

from src.ai.providers import Provider, DEFAULT_PROVIDER

load_dotenv()
logger = logging.getLogger(__name__)


def _sanitize_key(raw: str | None) -> str | None:
    """Trim common copy-paste artifacts from API keys.

    Handles leading "export ", surrounding quotes, and accidental trailing
    "export" suffixes that often sneak in when copying from shell history.
    """
    if not raw:
        return raw

    cleaned = raw.strip().strip("\"'")

    if cleaned.lower().startswith("export "):
        cleaned = cleaned.split(" ", 1)[1].strip()

    if cleaned.endswith("export"):
        cleaned = cleaned[: -len("export")].strip()

    return cleaned


class EmbeddingError(Exception):
    pass


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @property
    def provider(self) -> Provider:
        return Provider.GOOGLE


class GoogleEmbedder(EmbeddingProvider):
    """Google Gemini embedding provider with configurable dimensions.
    
    Model: gemini-embedding-001
    - Native dimension: 3072
    - Configurable: 128-3072 via output_dimensionality
    - Recommended: 768, 1536, 3072
    - Requires L2 normalization for dims < 3072
    
    Task types:
    - RETRIEVAL_DOCUMENT: For documents being indexed
    - RETRIEVAL_QUERY: For search queries  
    - SEMANTIC_SIMILARITY: For comparing text similarity
    - CLASSIFICATION: For categorization
    - CLUSTERING: For grouping similar items
    - QUESTION_ANSWERING: For Q&A systems
    - FACT_VERIFICATION: For fact-checking
    - CODE_RETRIEVAL_QUERY: For code search
    """
    
    VALID_TASK_TYPES = [
        "RETRIEVAL_DOCUMENT",
        "RETRIEVAL_QUERY", 
        "SEMANTIC_SIMILARITY",
        "CLASSIFICATION",
        "CLUSTERING",
        "QUESTION_ANSWERING",
        "FACT_VERIFICATION",
        "CODE_RETRIEVAL_QUERY",
    ]
    
    def __init__(self, model: str = "gemini-embedding-001", dimension: int = 768, task_type: str = "RETRIEVAL_DOCUMENT"):
        import google.generativeai as genai

        api_key = _sanitize_key(os.getenv("GEMINI_API_KEY"))
        if not api_key:
            raise EmbeddingError("GEMINI_API_KEY not found in environment")
        genai.configure(api_key=api_key)
        self._genai = genai
        self._model = model
        self._dimension = dimension
        self._task_type = task_type if task_type in self.VALID_TASK_TYPES else "RETRIEVAL_DOCUMENT"

    def embed_text(self, text: str, task_type: str = None) -> List[float]:
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")
        
        effective_task = task_type if task_type in self.VALID_TASK_TYPES else self._task_type
        
        resp = self._genai.embed_content(
            model=self._model,
            content=text,
            output_dimensionality=self._dimension,
            task_type=effective_task,
        )
        return resp["embedding"] if isinstance(resp, dict) else resp.embedding

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider(self) -> Provider:
        return Provider.GOOGLE


class OpenAIEmbedder(EmbeddingProvider):
    """OpenAI embedding provider with configurable dimensions.
    
    Models:
    - text-embedding-3-large: Native 3072d, configurable 256-3072
    - text-embedding-3-small: Native 1536d, configurable 256-1536  
    - text-embedding-ada-002: Fixed 1536d (legacy, no dimension param)
    """
    
    # Native dimensions per model (from OpenAI docs)
    MODEL_DIMS = {
        "text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(self, model: str = "text-embedding-3-large", dimension: int = None):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise EmbeddingError("openai package not installed; add to requirements.txt") from e

        api_key = _sanitize_key(os.getenv("OPENAI_API_KEY"))
        if not api_key:
            raise EmbeddingError("OPENAI_API_KEY not found in environment")
        self._client = OpenAI(api_key=api_key)
        self._model = model
        
        # Determine dimension: use provided, env, or native default
        native_dim = self.MODEL_DIMS.get(model, 1536)
        self._dimension = dimension or int(os.getenv("OPENAI_EMBEDDING_DIM", str(native_dim)))
        
        # ada-002 doesn't support dimension param
        self._supports_dimensions = "3-large" in model or "3-small" in model

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")
        
        params = {"model": self._model, "input": text}
        
        # Only add dimensions param for text-embedding-3 models
        if self._supports_dimensions:
            params["dimensions"] = self._dimension
            
        resp = self._client.embeddings.create(**params)
        return resp.data[0].embedding

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider(self) -> Provider:
        return Provider.OPENAI


def get_embedder(
    provider: Optional[Provider] = None, 
    model: Optional[str] = None, 
    dimension: Optional[int] = None,
    task_type: Optional[str] = None,
) -> EmbeddingProvider:
    """Factory function to get the appropriate embedder.
    
    Args:
        provider: google or openai (defaults to AI_PROVIDER env var)
        model: Model name (defaults to provider-specific env var)
        dimension: Output dimension (defaults to provider-specific env var or native)
        task_type: Task type for Gemini (ignored for OpenAI)
    
    Returns:
        Configured EmbeddingProvider instance
    """
    provider = provider or Provider(os.getenv("AI_PROVIDER", DEFAULT_PROVIDER.value))

    if provider == Provider.OPENAI:
        return OpenAIEmbedder(
            model=model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
            dimension=dimension,  # Will use env or native default if None
        )

    # Default: Google Gemini
    return GoogleEmbedder(
        model=model or os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
        dimension=dimension or int(os.getenv("GEMINI_EMBEDDING_DIM", "768")),
        task_type=task_type or os.getenv("GEMINI_TASK_TYPE", "RETRIEVAL_DOCUMENT"),
    )
