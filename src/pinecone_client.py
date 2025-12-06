"""
Pinecone client for vector operations

Optimized with:
- gRPC transport for better performance on data operations
- Connection pooling for parallel queries
- Cross-namespace search via query_namespaces
- Retry logic with exponential backoff
"""
import os
import time
from functools import wraps
from typing import List, Dict, Optional, Any

from dotenv import load_dotenv
import logging
from pinecone import ServerlessSpec

# Use gRPC client for better performance on upserts/queries
try:
    from pinecone.grpc import PineconeGRPC as Pinecone
    USING_GRPC = True
except ImportError:
    from pinecone import Pinecone
    USING_GRPC = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Configuration constants
DEFAULT_POOL_THREADS = 30  # Thread pool for parallel operations
DEFAULT_CONNECTION_POOL_SIZE = 30  # HTTP connection pool (non-gRPC only)
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds


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
                    error_str = str(e)
                    # Retry on transient errors (SSL, connection, timeout)
                    if any(err in error_str for err in ["SSL", "Connection", "Timeout", "timeout"]):
                        if attempt < max_retries - 1:
                            wait_time = delay * (2 ** attempt)  # Exponential backoff
                            logger.warning(f"Transient error in {func.__name__}, retrying in {wait_time}s ({attempt+1}/{max_retries}): {e}")
                            time.sleep(wait_time)
                            continue
                    raise e
            raise last_error
        return wrapper
    return decorator


class PineconeClient:
    """
    Optimized Pinecone client with:
    - gRPC transport for better data operation performance
    - Connection pooling for efficient parallel queries
    - Cross-namespace search capability
    - Automatic retry with exponential backoff
    """
    
    def __init__(self, pool_threads: int = DEFAULT_POOL_THREADS, index_name: Optional[str] = None, dimension: Optional[int] = None):
        """
        Initialize optimized Pinecone client.
        
        Args:
            pool_threads: Number of threads for parallel operations
        """
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY must be set in .env file")
        
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "transcripts")
        self.pool_threads = pool_threads
        
        # Initialize index with connection pooling for better parallel performance
        index_kwargs = {
            "name": self.index_name,
            "pool_threads": pool_threads,
        }
        # Add connection pool size for HTTP client (not needed for gRPC)
        if not USING_GRPC:
            index_kwargs["connection_pool_maxsize"] = DEFAULT_CONNECTION_POOL_SIZE
        
        self.index = self.pc.Index(**index_kwargs)
        
        transport = "gRPC" if USING_GRPC else "HTTP"
        logger.info(f"Pinecone client initialized [{transport}] for index: {self.index_name} (pool_threads={pool_threads})")

    def create_index(self, name: str, dimension: int, metric: str = "cosine", cloud: str = "aws", region: str = "us-east-1") -> bool:
        try:
            existing = [idx.name for idx in self.pc.list_indexes()]
            if name in existing:
                return True
            self.pc.create_index(
                name=name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
            logger.info(f"Created Pinecone index '{name}' ({dimension}d, {metric})")
            return True
        except Exception as e:
            logger.error(f"Error creating index '{name}': {e}")
            return False

    def list_index_dimensions(self) -> Dict[str, int]:
        dims = {}
        try:
            for idx in self.pc.list_indexes():
                try:
                    desc = self.pc.describe_index(idx.name)
                    dims[idx.name] = desc.dimension
                except Exception:
                    continue
        except Exception as e:
            logger.error(f"Error listing index dimensions: {e}")
        return dims

    @retry_on_error()
    def list_indexes(self) -> List[str]:
        """List all available Pinecone indexes."""
        try:
            indexes = self.pc.list_indexes()
            return [idx.name for idx in indexes]
        except Exception as e:
            logger.error(f"Error listing indexes: {e}")
            return [self.index_name]

    @retry_on_error()
    def list_namespaces(self) -> List[str]:
        """List all namespaces in the current index."""
        try:
            stats = self.index.describe_index_stats()
            return list(stats.namespaces.keys()) if stats.namespaces else [""]
        except Exception as e:
            logger.error(f"Error listing namespaces: {e}")
            return [""]

    def switch_index(self, index_name: str):
        """
        Switch to a different index with optimized settings.
        
        Args:
            index_name: Name of the index to switch to
        """
        self.index_name = index_name
        index_kwargs = {
            "name": index_name,
            "pool_threads": self.pool_threads,
        }
        if not USING_GRPC:
            index_kwargs["connection_pool_maxsize"] = DEFAULT_CONNECTION_POOL_SIZE
        
        self.index = self.pc.Index(**index_kwargs)
        logger.info(f"Switched to index: {index_name}")
    
    @retry_on_error()
    def query_similar(
        self, 
        query_embedding: List[float], 
        top_k: int = 5, 
        filter_dict: Optional[Dict] = None,
        namespace: str = ""
    ) -> List[Any]:
        """
        Find similar transcripts using embedding.
        
        Args:
            query_embedding: Vector embedding
            top_k: Number of results
            filter_dict: Optional metadata filters
            namespace: Namespace to query (default: "")
            
        Returns:
            List of matches with scores and metadata
        """
        try:
            query_params = {
                "vector": query_embedding,
                "top_k": top_k,
                "include_metadata": True,
                "namespace": namespace,
            }
            
            if filter_dict:
                query_params["filter"] = filter_dict
            
            results = self.index.query(**query_params)
            return results.matches
            
        except Exception as e:
            logger.error(f"Error querying similar vectors: {e}")
            return []
    
    def query_namespaces(
        self,
        query_embedding: List[float],
        namespaces: List[str],
        top_k: int = 5,
        filter_dict: Optional[Dict] = None,
        metric: str = "cosine",
        include_values: bool = False,
        include_metadata: bool = True,
        show_progress: bool = False,
    ) -> Any:
        """
        Query multiple namespaces in parallel and merge results.
        
        Uses Pinecone's built-in query_namespaces for efficient cross-namespace search.
        Results are merged and ranked by score, returning the top_k best matches.
        
        Args:
            query_embedding: Vector embedding to search with
            namespaces: List of namespaces to search (e.g., ['full_text', 'summaries'])
            top_k: Number of results to return (after merging)
            filter_dict: Optional metadata filters
            metric: Distance metric for ranking ('cosine', 'euclidean', 'dotproduct')
            include_values: Include vector values in response
            include_metadata: Include metadata in response
            show_progress: Show progress bar for queries
            
        Returns:
            Combined results with matches from all namespaces, ranked by score
        """
        try:
            query_params = {
                "vector": query_embedding,
                "namespaces": namespaces,
                "metric": metric,
                "top_k": top_k,
                "include_values": include_values,
                "include_metadata": include_metadata,
                "show_progress": show_progress,
            }
            
            if filter_dict:
                query_params["filter"] = filter_dict
            
            # Use Pinecone's built-in parallel cross-namespace query
            combined_results = self.index.query_namespaces(**query_params)
            
            logger.info(f"Cross-namespace query returned {len(combined_results.matches)} matches from {len(namespaces)} namespaces")
            return combined_results
            
        except Exception as e:
            logger.error(f"Error in cross-namespace query: {e}")
            # Fallback: query namespaces sequentially and merge manually
            return self._fallback_query_namespaces(
                query_embedding, namespaces, top_k, filter_dict, include_metadata
            )
    
    def _fallback_query_namespaces(
        self,
        query_embedding: List[float],
        namespaces: List[str],
        top_k: int,
        filter_dict: Optional[Dict],
        include_metadata: bool,
    ) -> Dict:
        """
        Fallback method for cross-namespace query if built-in method fails.
        Queries each namespace and merges results manually.
        """
        all_matches = []
        total_usage = {"read_units": 0}
        
        for ns in namespaces:
            matches = self.query_similar(
                query_embedding, 
                top_k=top_k, 
                filter_dict=filter_dict,
                namespace=ns
            )
            # Tag each match with its source namespace
            for match in matches:
                match.namespace = ns
            all_matches.extend(matches)
        
        # Sort by score (descending) and take top_k
        all_matches.sort(key=lambda x: x.score, reverse=True)
        merged_matches = all_matches[:top_k]
        
        # Create a response-like object
        class MergedResults:
            def __init__(self, matches, usage):
                self.matches = matches
                self.usage = usage
        
        return MergedResults(merged_matches, total_usage)

    @retry_on_error()
    def fetch_vectors(self, ids: List[str], namespace: str = "") -> Dict[str, Any]:
        """Fetch specific vectors by ids (small helper for existence checks)."""
        ns_param = namespace if namespace else None
        try:
            res = self.index.fetch(ids=ids, namespace=ns_param)
            return res.vectors if res else {}
        except Exception as e:
            logger.error(f"Error fetching vectors {ids}: {e}")
            return {}
    
    @retry_on_error()
    def get_all_vectors(self, namespace: str = "") -> List[Any]:
        """
        Get all vectors from index (for mind map generation).
        
        Uses batched fetching with retry logic for reliability.
        
        Args:
            namespace: Pinecone namespace to fetch from
            
        Returns:
            List of vectors with metadata
        """
        try:
            # Get index stats
            stats = self.index.describe_index_stats()
            logger.info(f"Index contains {stats.total_vector_count} vectors")
            
            all_vectors = []
            ids_to_fetch = []
            
            # List all IDs in namespace
            ns_param = namespace if namespace else None
            for batch in self.index.list(namespace=ns_param):
                ids_to_fetch.extend(batch)
            
            if not ids_to_fetch:
                logger.info(f"No vectors found in namespace '{namespace}'")
                return []
            
            # Fetch in optimized batches
            chunk_size = 100  # Pinecone recommended batch size
            for i in range(0, len(ids_to_fetch), chunk_size):
                chunk = ids_to_fetch[i:i+chunk_size]
                fetch_res = self._fetch_batch(chunk, ns_param)
                if fetch_res:
                    all_vectors.extend(fetch_res.vectors.values())
            
            logger.info(f"Fetched {len(all_vectors)} vectors from namespace '{namespace}'")
            return all_vectors
            
        except Exception as e:
            logger.error(f"Error getting vectors: {e}")
            return []
    
    @retry_on_error()
    def _fetch_batch(self, ids: List[str], namespace: Optional[str]) -> Any:
        """Fetch a batch of vectors with retry logic."""
        return self.index.fetch(ids=ids, namespace=namespace)
    
    @retry_on_error()
    def delete_vectors(self, ids: List[str], namespace: str = "") -> bool:
        """
        Delete specific vectors by ID.
        
        Args:
            ids: List of vector IDs to delete
            namespace: Namespace containing the vectors
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.index.delete(ids=ids, namespace=namespace if namespace else None)
            logger.info(f"Deleted {len(ids)} vectors from namespace '{namespace}'")
            return True
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            return False

    @retry_on_error()
    def delete_by_filter(self, flt: Dict, namespace: str = "") -> bool:
        """Delete vectors matching a metadata filter."""
        try:
            self.index.delete(filter=flt, namespace=namespace if namespace else None)
            logger.info(f"Deleted vectors by filter in namespace '{namespace}'")
            return True
        except Exception as e:
            logger.error(f"Error deleting by filter: {e}")
            return False

    @retry_on_error()
    def delete_all(self, namespace: str = "") -> bool:
        """Delete all vectors in a namespace."""
        try:
            self.index.delete(delete_all=True, namespace=namespace if namespace else None)
            logger.info(f"Deleted ALL vectors in namespace '{namespace}'")
            return True
        except Exception as e:
            logger.error(f"Error deleting all vectors: {e}")
            return False

    @retry_on_error()
    def update_metadata(self, vec_id: str, metadata: Dict, namespace: str = "") -> bool:
        """Update metadata for a vector."""
        try:
            self.index.update(id=vec_id, set_metadata=metadata, namespace=namespace if namespace else None)
            logger.info(f"Updated metadata for {vec_id} in namespace '{namespace}'")
            return True
        except Exception as e:
            logger.error(f"Error updating metadata: {e}")
            return False
    
    @retry_on_error()
    def get_index_info(self) -> Dict[str, Any]:
        """
        Get detailed index information including namespace stats.
        
        Returns:
            Dict with index metadata, stats, and status
        """
        try:
            stats = self.index.describe_index_stats()
            info = self.pc.describe_index(self.index_name)
            
            # Build namespace breakdown
            namespace_stats = {}
            if stats.namespaces:
                for ns_name, ns_info in stats.namespaces.items():
                    namespace_stats[ns_name] = {
                        "vector_count": ns_info.vector_count
                    }
            
            return {
                "name": self.index_name,
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "metric": info.metric,
                "host": info.host,
                "status": info.status,
                "namespaces": namespace_stats,
                "using_grpc": USING_GRPC,
            }
        except Exception as e:
            logger.error(f"Error getting index info: {e}")
            return {"error": str(e)}
    
    @retry_on_error()
    def upsert_vectors(
        self,
        vectors: List[Dict],
        namespace: str = "",
        batch_size: int = 100,
    ) -> int:
        """
        Upsert vectors with batching for optimal performance.
        
        Args:
            vectors: List of dicts with 'id', 'values', and optional 'metadata'
            namespace: Target namespace
            batch_size: Vectors per batch (default: 100)
            
        Returns:
            Number of vectors upserted
        """
        try:
            total_upserted = 0
            ns_param = namespace if namespace else None
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i+batch_size]
                self.index.upsert(vectors=batch, namespace=ns_param)
                total_upserted += len(batch)
            
            logger.info(f"Upserted {total_upserted} vectors to namespace '{namespace}'")
            return total_upserted
            
        except Exception as e:
            logger.error(f"Error upserting vectors: {e}")
            return 0
