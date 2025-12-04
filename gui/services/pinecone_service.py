from __future__ import annotations

from typing import List, Dict, Tuple, Any

from gui.services.clients import get_pinecone_client
from gui.state import state
from gui.utils.logging import log


def get_indexes_and_namespaces() -> Tuple[List[str], List[str], str, Dict[str, Any]]:
    """Return (indexes, namespaces, current_index, stats_dict)."""
    client = get_pinecone_client()
    indexes = client.list_indexes()
    namespaces = client.list_namespaces()
    current = client.index_name
    
    # Get detailed stats
    info = client.get_index_info()
    stats = {
        "vectors": info.get("total_vectors", 0),
        "dimension": info.get("dimension", "—"),
        "metric": info.get("metric", "—"),
        "namespaces": len(namespaces),
    }
    
    state.pinecone_indexes = indexes
    state.pinecone_stats = stats
    log('INFO', f"Found {len(indexes)} indexes, {len(namespaces)} namespaces, {stats['vectors']} vectors")
    return indexes, namespaces, current, stats


def switch_index(index_name: str) -> Tuple[List[str], Dict[str, Any]]:
    """Switch index and return (namespaces, stats)."""
    client = get_pinecone_client()
    client.switch_index(index_name)
    namespaces = client.list_namespaces()
    info = client.get_index_info()
    stats = {
        "vectors": info.get("total_vectors", 0),
        "dimension": info.get("dimension", "—"),
        "metric": info.get("metric", "—"),
        "namespaces": len(namespaces),
    }
    return namespaces, stats


def refresh_vectors(namespace: str = "") -> List[Dict]:
    client = get_pinecone_client()
    vectors = client.get_all_vectors(namespace)
    formatted = [_format_vector(vec) for vec in vectors]
    state.pinecone_vectors = formatted
    log('INFO', f"Loaded {len(formatted)} vectors from Pinecone")
    return formatted


def _format_vector(vec) -> Dict:
    meta = vec.metadata or {}
    title = meta.get('title') or meta.get('name') or 'Untitled'
    duration_ms = meta.get('duration_ms') or meta.get('duration') or 0
    minutes = int(duration_ms // 60000)
    seconds = int((duration_ms % 60000) // 1000)
    duration = f"{minutes}:{seconds:02d}" if duration_ms else '—'

    return {
        'id': vec.id,
        'short_id': f"{vec.id[:10]}…" if vec.id else '—',
        'title': title,
        'date': meta.get('date') or meta.get('start_at', '')[:10],
        'duration': duration,
        'tags': meta.get('themes') or meta.get('tags') or '—',
        'field_count': len(meta),
        'metadata': meta,
    }
