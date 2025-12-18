"""
Qdrant client for vector operations

Local-first vector storage with full visibility and control.
Drop-in replacement for PineconeClient with same interface.

Key advantages:
- Web dashboard at http://localhost:6333/dashboard
- See every vector, inspect payloads, debug visually
- Rich filtering (date ranges, topics, entities)
- No cloud dependency during development
- Your data stays on your machine
"""

import os
import time
import hashlib
import uuid
from functools import wraps
from typing import List, Dict, Optional, Any, Union
from datetime import datetime

import ssl

from dotenv import load_dotenv
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    ScoredPoint,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Range,
    UpdateStatus,
    CollectionStatus,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MAX_RETRIES = 3
RETRY_DELAY = 0.5


def _ensure_working_ca_bundle() -> None:
    """Ensure Python/httpx can create an SSL context.

    Why this exists:
    - On some macOS + Homebrew Python setups (especially with Python 3.13 and
      truststore/httpx), SSL context creation may fail early with
      `ssl.SSLError: [X509: NO_CERTIFICATE_OR_CRL_FOUND]` due to a broken CA
      bundle path.
    - Even when Qdrant is accessed over plain HTTP, httpx may initialize SSL
      plumbing during client construction.

    Strategy:
    - If the user's env already sets SSL_CERT_FILE, respect it.
    - Otherwise, try certifi; then Homebrew OpenSSL; then macOS system bundle.
    """

    if os.getenv("SSL_CERT_FILE"):
        return

    candidates: list[tuple[str, Optional[str]]] = []
    try:
        import certifi

        candidates.append((certifi.where(), None))
    except Exception:
        pass

    # Homebrew OpenSSL (common default for python.org/Homebrew builds)
    candidates.append(
        ("/opt/homebrew/etc/openssl@3/cert.pem", "/opt/homebrew/etc/openssl@3/certs")
    )

    # macOS system bundle
    candidates.append(("/etc/ssl/cert.pem", None))

    for cafile, capath in candidates:
        try:
            if cafile and os.path.exists(cafile) and os.path.getsize(cafile) > 0:
                ctx = ssl.create_default_context()
                ctx.load_verify_locations(cafile=cafile, capath=capath)
                os.environ.setdefault("SSL_CERT_FILE", cafile)
                if capath:
                    os.environ.setdefault("SSL_CERT_DIR", capath)
                return
        except Exception:
            continue


def string_to_uuid(s: str) -> str:
    """
    Convert a string ID to a deterministic UUID.

    Qdrant requires IDs to be either integers or UUIDs.
    This creates a consistent UUID from any string so we can
    use readable IDs like 'rec_abc123_chunk_0' while still
    satisfying Qdrant's requirements.

    The original string ID is stored in payload['_original_id']
    for reference.
    """
    # Create a deterministic UUID from the string using MD5 hash
    hash_bytes = hashlib.md5(s.encode()).digest()
    return str(uuid.UUID(bytes=hash_bytes))


def retry_on_error(max_retries: int = MAX_RETRIES, delay: float = RETRY_DELAY):
    """Decorator for retrying operations with exponential backoff."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (2**attempt)
                        logger.warning(
                            f"Error in {func.__name__}, retrying in {wait_time}s: {e}"
                        )
                        time.sleep(wait_time)
                        continue
                    raise e
            raise last_error

        return wrapper

    return decorator


class QdrantVectorClient:
    """
    Qdrant client matching PineconeClient interface for easy migration.

    Features:
    - Local-first with Docker
    - Web dashboard for visual debugging
    - Rich payload storage (no 40KB limit)
    - Temporal filtering built-in
    - Full visibility into your data
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize Qdrant client.

        Connection priority:
        1. Explicit url/api_key parameters
        2. QDRANT_URL + QDRANT_API_KEY env vars (for cloud)
        3. Default to localhost:6333 (local Docker)
        """
        _ensure_working_ca_bundle()

        # Determine connection settings
        url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = api_key or os.getenv("QDRANT_API_KEY")

        # Initialize client
        if api_key:
            self.client = QdrantClient(url=url, api_key=api_key)
            logger.info(f"Qdrant client connected to cloud: {url}")
        else:
            self.client = QdrantClient(url=url)
            logger.info(f"Qdrant client connected to local: {url}")

        self.collection_name = collection_name or os.getenv(
            "QDRANT_COLLECTION", "transcripts"
        )
        self.url = url

        # Store connection info for dashboard access
        self._dashboard_url = (
            f"{url.rstrip('/')}/dashboard" if "localhost" in url else None
        )

    @property
    def dashboard_url(self) -> Optional[str]:
        """URL to Qdrant web dashboard (local only)."""
        return self._dashboard_url

    @property
    def index_name(self) -> str:
        """Alias for collection_name (Pinecone compatibility)."""
        return self.collection_name

    # ---------------------------------------------------------------------
    # Pinecone SDK compatibility layer
    # ---------------------------------------------------------------------
    # The GUI historically used the Pinecone SDK shape: client.index.query(...),
    # client.index.fetch(...), client.index.describe_index_stats(),
    # and client.pc.describe_index(...). When VECTOR_DB=qdrant we still want
    # those screens/actions to work without rewriting the entire GUI.

    class _CompatMatch:
        def __init__(
            self,
            _id: str,
            score: float = 0.0,
            metadata: Optional[Dict[str, Any]] = None,
            namespace: str = "",
        ):
            self.id = _id
            self.score = score
            self.metadata = metadata or {}
            self.namespace = namespace

    class _CompatQueryResults:
        def __init__(self, matches: List["QdrantVectorClient._CompatMatch"]):
            self.matches = matches

    class _CompatVector:
        def __init__(
            self,
            _id: str,
            values: Optional[List[float]] = None,
            metadata: Optional[Dict[str, Any]] = None,
        ):
            self.id = _id
            self.values = values
            self.metadata = metadata or {}

    class _CompatFetchResults:
        def __init__(self, vectors: Dict[str, "QdrantVectorClient._CompatVector"]):
            self.vectors = vectors

    class _CompatPagination:
        def __init__(self, nxt: Optional[str] = None):
            self.next = nxt

    class _CompatFetchByMetadataResults:
        def __init__(
            self,
            vectors: Dict[str, "QdrantVectorClient._CompatVector"],
            pagination: Optional["QdrantVectorClient._CompatPagination"] = None,
        ):
            self.vectors = vectors
            self.pagination = pagination

    class _CompatNamespaceStats:
        def __init__(self, vector_count: int = 0):
            self.vector_count = vector_count

    class _CompatIndexStats:
        def __init__(
            self,
            dimension: int,
            namespaces: Dict[str, "QdrantVectorClient._CompatNamespaceStats"],
            total_vector_count: int = 0,
            metric: str = "cosine",
        ):
            self.dimension = dimension
            self.namespaces = namespaces
            self.total_vector_count = total_vector_count
            self.metric = metric

    class _PineconeIndexCompat:
        """Provides a Pinecone-like .index surface on top of Qdrant."""

        def __init__(self, parent: "QdrantVectorClient"):
            self._p = parent

        def query(
            self,
            vector: Optional[List[float]] = None,
            id: Optional[str] = None,
            top_k: int = 10,
            namespace: Optional[str] = None,
            filter: Optional[Dict[str, Any]] = None,
            include_metadata: bool = True,
            include_values: bool = False,
            **kwargs,
        ) -> "QdrantVectorClient._CompatQueryResults":
            # Pinecone supports querying by id; for Qdrant we fetch the vector then query by its values.
            if id and vector is None:
                fetched = self.fetch(ids=[id], namespace=namespace)
                v = fetched.vectors.get(id)
                if not v or not v.values:
                    raise ValueError(
                        "Vector id not found (or values unavailable) for id-based query"
                    )
                vector = v.values

            if vector is None:
                raise ValueError("Missing query vector")

            ns = namespace or ""
            scored = self._p.query_similar(
                query_embedding=vector,
                top_k=top_k,
                filter_dict=filter,
                namespace=ns,
            )
            matches: List[QdrantVectorClient._CompatMatch] = []
            for sp in scored:
                # `query_similar()` returns Pinecone-like match objects when
                # VECTOR_DB=qdrant. Older code paths may still pass raw
                # ScoredPoint objects; handle both shapes.
                payload = getattr(sp, "payload", None)
                if payload is None:
                    meta_in = getattr(sp, "metadata", None) or {}
                    original_id = str(getattr(sp, "id", ""))
                    score = float(getattr(sp, "score", 0.0) or 0.0)
                else:
                    meta_in = payload or {}
                    original_id = meta_in.get("_original_id") or str(
                        getattr(sp, "id", "")
                    )
                    score = float(getattr(sp, "score", 0.0) or 0.0)

                meta = meta_in if include_metadata else {}
                matches.append(
                    QdrantVectorClient._CompatMatch(
                        original_id, score=score, metadata=meta, namespace=ns
                    )
                )
            return QdrantVectorClient._CompatQueryResults(matches)

        def fetch(
            self, ids: List[str], namespace: Optional[str] = None, **kwargs
        ) -> "QdrantVectorClient._CompatFetchResults":
            # Namespace is stored in payload; we ignore namespace here and return whatever matches ids.
            raw = self._p.fetch_vectors(ids, namespace=namespace or "")
            vectors: Dict[str, QdrantVectorClient._CompatVector] = {}
            for original_id, data in raw.items():
                vectors[original_id] = QdrantVectorClient._CompatVector(
                    _id=original_id,
                    values=data.get("values"),
                    metadata=data.get("metadata") or {},
                )
            return QdrantVectorClient._CompatFetchResults(vectors)

        def fetch_by_metadata(
            self,
            filter: Dict[str, Any],
            namespace: Optional[str] = None,
            limit: int = 100,
            pagination_token: Optional[str] = None,
            **kwargs,
        ) -> "QdrantVectorClient._CompatFetchByMetadataResults":
            # Qdrant scroll doesn't support Pinecone-style pagination tokens in this shim.
            _ = pagination_token
            ns = namespace or ""
            rows = self._p.fetch_by_metadata(
                filter_dict=filter, namespace=ns, limit=limit
            )
            vectors: Dict[str, QdrantVectorClient._CompatVector] = {}
            for r in rows:
                meta = r.get("metadata") or {}
                original_id = meta.get("_original_id") or str(r.get("id"))
                vectors[original_id] = QdrantVectorClient._CompatVector(
                    _id=original_id,
                    values=r.get("values"),
                    metadata=meta,
                )
            return QdrantVectorClient._CompatFetchByMetadataResults(
                vectors=vectors, pagination=None
            )

        def describe_index_stats(self) -> "QdrantVectorClient._CompatIndexStats":
            info = self._p.get_collection_info() or {}
            dim = int(info.get("dimension") or 0)
            metric = str(info.get("metric") or "cosine")
            total = int(info.get("total_vectors") or 0)
            # Qdrant doesn't provide namespace counts cheaply; report zero counts.
            ns_names = list((info.get("namespaces") or {}).keys())
            namespaces = {
                n: QdrantVectorClient._CompatNamespaceStats(0) for n in ns_names
            }
            return QdrantVectorClient._CompatIndexStats(
                dimension=dim,
                namespaces=namespaces,
                total_vector_count=total,
                metric=metric,
            )

        def list_namespaces(self) -> List[str]:
            return self._p.list_namespaces()

        def query_namespaces(
            self,
            vector: List[float],
            namespaces: List[str],
            top_k: int = 10,
            filter: Optional[Dict[str, Any]] = None,
            include_metadata: bool = True,
            metric: str = "cosine",
            **kwargs,
        ) -> Any:
            _ = include_metadata
            _ = metric
            res = self._p.query_namespaces(
                query_embedding=vector,
                namespaces=namespaces,
                top_k=top_k,
                filter_dict=filter,
            )
            # Ensure returned matches have id/metadata/namespace fields.
            matches: List[QdrantVectorClient._CompatMatch] = []
            if hasattr(res, "matches"):
                for m in res.matches:
                    meta = (
                        getattr(m, "payload", None)
                        or getattr(m, "metadata", None)
                        or {}
                    )
                    original_id = meta.get("_original_id") or str(getattr(m, "id", ""))
                    matches.append(
                        QdrantVectorClient._CompatMatch(
                            original_id,
                            score=float(getattr(m, "score", 0.0) or 0.0),
                            metadata=meta,
                            namespace=str(getattr(m, "namespace", "") or ""),
                        )
                    )
            return QdrantVectorClient._CompatQueryResults(matches)

        def upsert(
            self, vectors: List[Any], namespace: Optional[str] = None, **kwargs
        ) -> bool:
            # Pinecone accepts list of tuples (id, values, metadata) or list of dicts.
            ns = namespace or ""
            normalized: List[Dict[str, Any]] = []
            for v in vectors:
                if isinstance(v, dict):
                    normalized.append(
                        {
                            "id": v.get("id"),
                            "values": v.get("values"),
                            "metadata": v.get("metadata") or {},
                        }
                    )
                else:
                    # tuple/list: (id, values, metadata)
                    vec_id = v[0]
                    values = v[1]
                    meta = v[2] if len(v) > 2 else {}
                    normalized.append(
                        {"id": vec_id, "values": values, "metadata": meta or {}}
                    )
            return bool(self._p.upsert_vectors(normalized, namespace=ns))

        def delete(
            self,
            ids: Optional[List[str]] = None,
            delete_all: bool = False,
            namespace: Optional[str] = None,
            filter: Optional[Dict[str, Any]] = None,
            **kwargs,
        ) -> bool:
            ns = namespace or ""
            if delete_all:
                return bool(self._p.delete_all(namespace=ns))
            if filter is not None:
                return bool(self._p.delete_by_filter(filter, namespace=ns))
            if ids:
                return bool(self._p.delete_vectors(ids, namespace=ns))
            return False

    class _PineconeClientCompat:
        """Provides a minimal Pinecone-like .pc surface."""

        def __init__(self, parent: "QdrantVectorClient"):
            self._p = parent

        def describe_index(self, name: str) -> Any:
            _ = name
            info = self._p.get_collection_info() or {}
            metric = info.get("metric") or "cosine"
            return type("IndexInfo", (), {"metric": metric})()

    @property
    def index(self) -> "QdrantVectorClient._PineconeIndexCompat":
        """Pinecone SDK-compatible index handle (client.index.*)."""
        return QdrantVectorClient._PineconeIndexCompat(self)

    @property
    def pc(self) -> "QdrantVectorClient._PineconeClientCompat":
        """Pinecone SDK-compatible client handle (client.pc.*)."""
        return QdrantVectorClient._PineconeClientCompat(self)

    # =========================================================================
    # COLLECTION MANAGEMENT (= Pinecone Index management)
    # =========================================================================

    def create_collection(
        self, name: str, dimension: int, metric: str = "cosine"
    ) -> bool:
        """
        Create a new collection (equivalent to Pinecone create_index).

        Args:
            name: Collection name
            dimension: Vector dimension (must match your embedding model)
            metric: Distance metric (cosine, euclidean, dotproduct)
        """
        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dotproduct": Distance.DOT,
        }
        try:
            # Check if exists
            collections = self.list_collections()
            if name in collections:
                logger.info(f"Collection '{name}' already exists")
                return True

            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=distance_map.get(metric, Distance.COSINE),
                ),
            )
            logger.info(f"Created Qdrant collection '{name}' ({dimension}d, {metric})")
            return True
        except Exception as e:
            logger.error(f"Error creating collection '{name}': {e}")
            return False

    def create_index(
        self, name: str, dimension: int, metric: str = "cosine", **kwargs
    ) -> bool:
        """Alias for create_collection (Pinecone compatibility)."""
        return self.create_collection(name, dimension, metric)

    @retry_on_error()
    def list_collections(self) -> List[str]:
        """List all collections (equivalent to Pinecone list_indexes)."""
        try:
            collections = self.client.get_collections().collections
            return [c.name for c in collections]
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []

    def list_indexes(self) -> List[str]:
        """Alias for list_collections (Pinecone compatibility)."""
        return self.list_collections()

    def list_index_dimensions(self) -> Dict[str, int]:
        """Get dimensions for all collections."""
        dims = {}
        try:
            for name in self.list_collections():
                info = self.client.get_collection(name)
                dims[name] = info.config.params.vectors.size
        except Exception as e:
            logger.error(f"Error getting collection dimensions: {e}")
        return dims

    @retry_on_error()
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get detailed collection stats (equivalent to Pinecone describe_index_stats).

        Returns dict with:
        - name, total_vectors, dimension, metric, status
        - namespaces (unique namespace values in payloads)
        """
        try:
            info = self.client.get_collection(self.collection_name)

            # Get namespace breakdown by scrolling (for small collections)
            # For large collections, this could be expensive
            namespaces = set()
            try:
                # Quick sample to find namespaces
                results, _ = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    with_payload=["namespace"],
                )
                for point in results:
                    if point.payload and "namespace" in point.payload:
                        namespaces.add(point.payload["namespace"])
            except Exception:
                pass

            return {
                "name": self.collection_name,
                "total_vectors": info.points_count,
                "dimension": info.config.params.vectors.size,
                "metric": str(info.config.params.vectors.distance).lower(),
                "status": str(info.status),
                "namespaces": {ns: {} for ns in namespaces} if namespaces else {},
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}

    def get_index_info(self) -> Dict[str, Any]:
        """Alias for get_collection_info (Pinecone compatibility)."""
        return self.get_collection_info()

    def switch_collection(self, collection_name: str):
        """Switch to a different collection."""
        self.collection_name = collection_name
        logger.info(f"Switched to collection: {collection_name}")

    def switch_index(self, index_name: str):
        """Alias for switch_collection (Pinecone compatibility)."""
        self.switch_collection(index_name)

    def list_namespaces(self) -> List[str]:
        """
        List all unique namespace values in the collection.

        Note: Qdrant doesn't have native namespaces like Pinecone.
        We store namespace as a payload field and query for unique values.
        """
        try:
            info = self.get_collection_info()
            namespaces = list(info.get("namespaces", {}).keys())
            return namespaces if namespaces else [""]
        except Exception as e:
            logger.error(f"Error listing namespaces: {e}")
            return [""]

    # =========================================================================
    # VECTOR OPERATIONS
    # =========================================================================

    @retry_on_error()
    def upsert_vectors(
        self,
        vectors: List[Dict[str, Any]],
        namespace: str = "",
    ) -> bool:
        """
        Upsert vectors with metadata.

        Args:
            vectors: List of dicts with 'id', 'values', and 'metadata'
            namespace: Namespace to store in (stored in payload)

        Example:
            client.upsert_vectors([
                {
                    "id": "rec_abc123_chunk_0",
                    "values": [0.1, 0.2, ...],
                    "metadata": {
                        "recording_id": "abc123",
                        "text": "...",
                        "start_at": "2024-12-15T10:30:00Z",
                    }
                }
            ], namespace="full_text")
        """
        points = []
        for vec in vectors:
            payload = vec.get("metadata", {}).copy()
            # Add namespace to payload for filtering
            if namespace:
                payload["namespace"] = namespace

            # Convert string ID to UUID (Qdrant requires UUID or int)
            original_id = vec["id"]
            qdrant_id = string_to_uuid(original_id)
            # Store original ID in payload for reference
            payload["_original_id"] = original_id

            points.append(
                PointStruct(
                    id=qdrant_id,
                    vector=vec["values"],
                    payload=payload,
                )
            )

        result = self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        success = result.status == UpdateStatus.COMPLETED
        if success:
            logger.info(
                f"Upserted {len(points)} vectors to '{self.collection_name}' (namespace={namespace or 'default'})"
            )
        return success

    def upsert(self, vectors: List[Dict], namespace: str = "") -> bool:
        """Alias for upsert_vectors (Pinecone compatibility)."""
        return self.upsert_vectors(vectors, namespace)

    def _scored_point_to_match(
        self, point: Any, namespace_fallback: str = ""
    ) -> "QdrantVectorClient._CompatMatch":
        """Normalize a Qdrant result into a Pinecone-like match object.

        Many parts of the GUI/services assume `match.metadata` exists. Qdrant's
        `ScoredPoint` uses `.payload` instead. This helper keeps the rest of the
        codebase provider-neutral.
        """
        payload = getattr(point, "payload", None) or {}
        original_id = payload.get("_original_id") or str(getattr(point, "id", ""))
        score = float(getattr(point, "score", 0.0) or 0.0)
        ns = payload.get("namespace") or namespace_fallback or ""
        return QdrantVectorClient._CompatMatch(
            original_id,
            score=score,
            metadata=payload,
            namespace=str(ns),
        )

    @retry_on_error()
    def query_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict] = None,
        namespace: str = "",
    ) -> List["QdrantVectorClient._CompatMatch"]:
        """
        Find similar vectors by embedding.

        Args:
            query_embedding: Vector to search with
            top_k: Number of results
            filter_dict: Metadata filters (optional)
            namespace: Namespace to search in (optional)

        Returns:
            List of Pinecone-like match objects with `.id`, `.score`, `.metadata`,
            and `.namespace`.
        """
        # Build filter conditions
        conditions = []

        # Namespace filter
        if namespace:
            conditions.append(
                FieldCondition(key="namespace", match=MatchValue(value=namespace))
            )

        # Additional filters from filter_dict
        if filter_dict:
            for key, value in filter_dict.items():
                if isinstance(value, list):
                    conditions.append(
                        FieldCondition(key=key, match=MatchAny(any=value))
                    )
                elif isinstance(value, dict):
                    # Handle range filters like {"$gte": "2024-01-01"}
                    range_params = {}
                    if "$gte" in value:
                        range_params["gte"] = value["$gte"]
                    if "$lte" in value:
                        range_params["lte"] = value["$lte"]
                    if "$gt" in value:
                        range_params["gt"] = value["$gt"]
                    if "$lt" in value:
                        range_params["lt"] = value["$lt"]
                    if range_params:
                        conditions.append(
                            FieldCondition(key=key, range=Range(**range_params))
                        )
                else:
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )

        query_filter = Filter(must=conditions) if conditions else None

        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            ).points
            return [
                self._scored_point_to_match(p, namespace_fallback=namespace)
                for p in results
            ]
        except Exception as e:
            logger.error(f"Error querying vectors: {e}")
            return []

    def query_namespaces(
        self,
        query_embedding: List[float],
        namespaces: List[str],
        top_k: int = 5,
        filter_dict: Optional[Dict] = None,
        **kwargs,
    ) -> Any:
        """
        Query multiple namespaces and merge results.

        Since Qdrant uses payload-based namespaces, we can do this
        in a single query with an OR filter.
        """
        # Build namespace filter
        conditions = []
        if namespaces:
            conditions.append(
                FieldCondition(key="namespace", match=MatchAny(any=namespaces))
            )

        if filter_dict:
            for key, value in filter_dict.items():
                if isinstance(value, list):
                    conditions.append(
                        FieldCondition(key=key, match=MatchAny(any=value))
                    )
                else:
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )

        query_filter = Filter(must=conditions) if conditions else None

        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            ).points

            # Format like Pinecone response: `.matches` contains match objects
            # with `.metadata` (payload) and `.namespace` fields.
            class NamespaceResults:
                def __init__(self, matches):
                    self.matches = matches
                    self.usage = {"read_units": len(matches)}

            matches = [self._scored_point_to_match(p) for p in results]
            return NamespaceResults(matches)
        except Exception as e:
            logger.error(f"Error in namespace query: {e}")
            return type("EmptyResults", (), {"matches": [], "usage": {}})()

    @retry_on_error()
    def get_all_vectors(self, namespace: str = "", limit: int = 1000) -> List[Dict]:
        """
        Retrieve all vectors from collection/namespace.

        Returns list of dicts with id, values, and metadata.
        """
        conditions = []
        if namespace:
            conditions.append(
                FieldCondition(key="namespace", match=MatchValue(value=namespace))
            )

        query_filter = Filter(must=conditions) if conditions else None

        try:
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                scroll_filter=query_filter,
                with_vectors=True,
                with_payload=True,
            )

            vectors = []
            for point in results:
                # Return original ID from payload if available
                original_id = (
                    point.payload.get("_original_id", str(point.id))
                    if point.payload
                    else str(point.id)
                )
                vectors.append(
                    {
                        "id": original_id,
                        "values": point.vector,
                        "metadata": point.payload or {},
                    }
                )
            return vectors
        except Exception as e:
            logger.error(f"Error getting all vectors: {e}")
            return []

    @retry_on_error()
    def fetch_vectors(self, ids: List[str], namespace: str = "") -> Dict[str, Any]:
        """Fetch specific vectors by ID."""
        try:
            # Convert string IDs to UUIDs
            qdrant_ids = [string_to_uuid(id) for id in ids]
            results = self.client.retrieve(
                collection_name=self.collection_name,
                ids=qdrant_ids,
                with_vectors=True,
                with_payload=True,
            )
            # Return with original IDs from payload
            output = {}
            for p in results:
                original_id = p.payload.get("_original_id", str(p.id))
                output[original_id] = {"values": p.vector, "metadata": p.payload}
            return output
        except Exception as e:
            logger.error(f"Error fetching vectors: {e}")
            return {}

    # =========================================================================
    # DELETE OPERATIONS
    # =========================================================================

    @retry_on_error()
    def delete_vectors(self, ids: List[str], namespace: str = "") -> bool:
        """Delete vectors by ID (converts string IDs to UUIDs)."""
        try:
            # Convert string IDs to UUIDs
            qdrant_ids = [string_to_uuid(id) for id in ids]
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qdrant_ids,
            )
            logger.info(f"Deleted {len(ids)} vectors from '{self.collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            return False

    @retry_on_error()
    def delete_by_filter(self, filter_dict: Dict, namespace: str = "") -> bool:
        """Delete vectors matching filter."""
        conditions = []
        if namespace:
            conditions.append(
                FieldCondition(key="namespace", match=MatchValue(value=namespace))
            )
        for key, value in filter_dict.items():
            if isinstance(value, list):
                conditions.append(FieldCondition(key=key, match=MatchAny(any=value)))
            else:
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(must=conditions),
            )
            logger.info(f"Deleted vectors by filter from '{self.collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Error deleting by filter: {e}")
            return False

    @retry_on_error()
    def delete_all(self, namespace: str = "") -> bool:
        """Delete all vectors (optionally in a namespace)."""
        try:
            if namespace:
                return self.delete_by_filter({"namespace": namespace})
            else:
                # Delete entire collection and recreate
                info = self.get_collection_info()
                dimension = info.get("dimension", 768)
                metric = info.get("metric", "cosine")
                self.client.delete_collection(self.collection_name)
                self.create_collection(self.collection_name, dimension, metric)
                logger.info(f"Deleted all vectors from '{self.collection_name}'")
                return True
        except Exception as e:
            logger.error(f"Error deleting all: {e}")
            return False

    # =========================================================================
    # METADATA OPERATIONS
    # =========================================================================

    @retry_on_error()
    def update_metadata(
        self, vec_id: str, metadata: Dict[str, Any], namespace: str = ""
    ) -> bool:
        """Update metadata for a specific vector."""
        try:
            self.client.set_payload(
                collection_name=self.collection_name,
                payload=metadata,
                points=[vec_id],
            )
            return True
        except Exception as e:
            logger.error(f"Error updating metadata: {e}")
            return False

    @retry_on_error()
    def fetch_by_metadata(
        self,
        filter_dict: Dict,
        namespace: str = "",
        limit: int = 100,
    ) -> List[Dict]:
        """
        Fetch vectors by metadata filter.

        Great for finding vectors by recording_id, date range, etc.
        """
        conditions = []
        if namespace:
            conditions.append(
                FieldCondition(key="namespace", match=MatchValue(value=namespace))
            )

        for key, value in filter_dict.items():
            if isinstance(value, list):
                conditions.append(FieldCondition(key=key, match=MatchAny(any=value)))
            elif isinstance(value, dict):
                # Handle range filters
                range_params = {}
                for op, val in value.items():
                    if op in ("$gte", "gte"):
                        range_params["gte"] = val
                    elif op in ("$lte", "lte"):
                        range_params["lte"] = val
                    elif op in ("$gt", "gt"):
                        range_params["gt"] = val
                    elif op in ("$lt", "lt"):
                        range_params["lt"] = val
                if range_params:
                    conditions.append(
                        FieldCondition(key=key, range=Range(**range_params))
                    )
            else:
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

        query_filter = Filter(must=conditions) if conditions else None

        try:
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                scroll_filter=query_filter,
                with_vectors=True,
                with_payload=True,
            )
            return [
                {"id": p.id, "values": p.vector, "metadata": p.payload} for p in results
            ]
        except Exception as e:
            logger.error(f"Error fetching by metadata: {e}")
            return []

    # =========================================================================
    # TEMPORAL QUERIES (New for life logging!)
    # =========================================================================

    def query_by_date_range(
        self,
        query_embedding: List[float],
        start_date: str,
        end_date: str,
        top_k: int = 10,
        namespace: str = "",
    ) -> List["QdrantVectorClient._CompatMatch"]:
        """
        Search within a specific date range.

        Args:
            query_embedding: Search vector
            start_date: ISO date string (e.g., "2024-01-01")
            end_date: ISO date string (e.g., "2024-01-31")
            top_k: Number of results
            namespace: Namespace to search

        This is the kind of query that's awkward in Pinecone but natural in Qdrant.
        """
        filter_dict = {"start_at": {"$gte": start_date, "$lte": end_date}}
        return self.query_similar(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_dict=filter_dict,
            namespace=namespace,
        )

    def get_recordings_by_date(self, date: str, namespace: str = "") -> List[Dict]:
        """
        Get all recordings from a specific date.

        Args:
            date: ISO date string (e.g., "2024-12-15")
            namespace: Namespace to search
        """
        return self.fetch_by_metadata(
            filter_dict={"start_at": {"$gte": date, "$lt": f"{date}T23:59:59"}},
            namespace=namespace,
        )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def health_check(self) -> bool:
        """Check if Qdrant is reachable."""
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive stats for display in UI."""
        try:
            info = self.get_collection_info()
            return {
                "connected": True,
                "url": self.url,
                "collection": self.collection_name,
                "vectors": info.get("total_vectors", 0),
                "dimension": info.get("dimension", "—"),
                "metric": info.get("metric", "—"),
                "status": info.get("status", "unknown"),
                "dashboard": self.dashboard_url,
            }
        except Exception as e:
            return {
                "connected": False,
                "error": str(e),
            }


# Factory function for easy client creation
def get_qdrant_client(collection_name: Optional[str] = None) -> QdrantVectorClient:
    """Create a Qdrant client instance."""
    return QdrantVectorClient(collection_name=collection_name)
