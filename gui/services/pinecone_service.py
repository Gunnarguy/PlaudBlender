from __future__ import annotations

from typing import List, Dict, Tuple, Any, Optional

import os
from gui.services.clients import get_pinecone_client
from gui.services.embedding_service import get_embedding_service
from gui.services import transcripts_service, settings_service
from gui.state import state
from gui.utils.logging import log


def get_indexes_and_namespaces() -> Tuple[List[str], List[str], str, Dict[str, Any]]:
    """Return (indexes, namespaces, current_index, stats_dict)."""
    embedder = get_embedding_service()
    target_dim = embedder.dimension
    provider_name = getattr(embedder, 'provider', None)
    client = get_pinecone_client()
    indexes = client.list_indexes()
    log('INFO', f"Pinecone indexes: {indexes}")
    raw_namespaces = client.list_namespaces()
    current = client.index_name

    info = client.get_index_info()
    ns_from_info = list(info.get("namespaces", {}).keys()) if isinstance(info, dict) else []
    namespaces = ns_from_info or [ns for ns in raw_namespaces if ns is not None]

    stats = {
        "vectors": info.get("total_vectors", 0) if isinstance(info, dict) else 0,
        "dimension": info.get("dimension", "—") if isinstance(info, dict) else "—",
        "metric": info.get("metric", "—") if isinstance(info, dict) else "—",
        "namespaces": len(namespaces),
        "namespace_names": namespaces,
        "target_dim": target_dim,
        "dim_mismatch": (info.get("dimension") != target_dim) if isinstance(info, dict) else False,
        "provider": provider_name.value if provider_name else "google",
        "index_name": info.get("name") if isinstance(info, dict) else client.index_name,
    }

    state.pinecone_indexes = indexes
    state.pinecone_stats = stats
    log('INFO', f"Found {len(indexes)} indexes, {len(namespaces)} namespaces, {stats['vectors']} vectors; "
                f"current index={stats.get('index_name')}, dim={stats.get('dimension')}, target_dim={target_dim}")
    return indexes, namespaces, current, stats


def switch_index(index_name: str) -> Tuple[List[str], Dict[str, Any]]:
    """Switch index and return (namespaces, stats)."""
    client = get_pinecone_client()
    log('INFO', f"Switching Pinecone index -> {index_name}")
    client.switch_index(index_name)
    _persist_index_choice(index_name)
    namespaces = client.list_namespaces()
    info = client.get_index_info()
    embedder = get_embedding_service()
    target_dim = embedder.dimension
    provider_name = getattr(embedder, 'provider', None)
    stats = {
        "vectors": info.get("total_vectors", 0),
        "dimension": info.get("dimension", "—"),
        "metric": info.get("metric", "—"),
        "namespaces": len(namespaces),
        "index_name": info.get("name", index_name),
        "target_dim": target_dim,
        "dim_mismatch": (info.get("dimension") != target_dim),
        "provider": provider_name.value if provider_name else "google",
    }
    log('INFO', f"Switched to {stats['index_name']} (dim={stats['dimension']}, target={target_dim}, "
                f"namespaces={len(namespaces)}, vectors={stats['vectors']})")
    return namespaces, stats


def refresh_vectors(namespace: str = "") -> List[Dict]:
    client = get_pinecone_client()
    if not namespace:
        fallback_ns = state.pinecone_stats.get('namespace_names') if state.pinecone_stats else None
        namespace = (fallback_ns or [""])[0] if isinstance(fallback_ns, list) and fallback_ns else ""
    vectors = client.get_all_vectors(namespace)
    formatted = [_format_vector(vec) for vec in vectors]
    # Sort by date descending if available
    formatted.sort(key=lambda v: v.get('date', ''), reverse=True)
    state.pinecone_vectors = formatted
    log('INFO', f"Loaded {len(formatted)} vectors from Pinecone")
    return formatted


def update_vector_metadata(vec_id: str, metadata: Dict[str, Any], namespace: str = "") -> bool:
    """Update metadata for a vector and return success."""
    client = get_pinecone_client()
    return client.update_metadata(vec_id, metadata, namespace=namespace)


def delete_vectors(ids: Optional[List[str]] = None, flt: Optional[Dict] = None, delete_all: bool = False, namespace: str = "") -> str:
    """Delete vectors by ids, filter, or delete_all. Returns a status string."""
    client = get_pinecone_client()
    if delete_all:
        ok = client.delete_all(namespace=namespace)
        return "all" if ok else "error"
    if flt is not None:
        ok = client.delete_by_filter(flt, namespace=namespace)
        return "filtered" if ok else "error"
    ids = ids or []
    ok = client.delete_vectors(ids, namespace=namespace)
    return str(len(ids)) if ok else "error"


def ensure_matching_index(target_index: str, dimension: int) -> str:
    """Ensure an index exists with the given name and dimension, return the index name used."""
    client = get_pinecone_client()
    indexes = client.list_indexes()
    dims = getattr(client, 'list_index_dimensions', lambda: {})()
    log('INFO', f"Ensuring matching index '{target_index}' at dim {dimension}; existing dims: {dims}")
    if target_index in dims and dims[target_index] != dimension:
        raise ValueError(
            f"Index '{target_index}' has dimension {dims[target_index]}, expected {dimension}."
        )
    if target_index not in indexes:
        created = client.create_index(name=target_index, dimension=dimension)
        if not created:
            raise RuntimeError(f"Failed to create index {target_index}")
        log('INFO', f"Created index '{target_index}' at dim {dimension}")
    client.switch_index(target_index)
    _persist_index_choice(target_index)
    return target_index


def find_matching_index(dimension: int) -> Optional[str]:
    """Return an existing index name that matches dimension, if any."""
    client = get_pinecone_client()
    dims = getattr(client, 'list_index_dimensions', lambda: {})()
    for name, dim in dims.items():
        if dim == dimension:
            return name
    return None


def reembed_all_into_index(index_name: str, namespace: str = "") -> None:
    """Re-embed all Plaud recordings into the target index/namespace."""
    from src.models.vector_metadata import build_metadata, compute_text_hash

    log('INFO', f"Re-embed start -> index={index_name}, namespace={namespace or 'full_text'}")
    embedder = get_embedding_service()
    target_ns = namespace or "full_text"
    ensure_matching_index(index_name, embedder.dimension)

    recordings = transcripts_service.fetch_transcripts()
    total = len(recordings)
    log('INFO', f"Re-embed fetched {total} recordings")
    if not recordings:
        return {
            "index": index_name,
            "namespace": target_ns,
            "upserted": 0,
            "failed": 0,
            "total": 0,
        }

    client = get_pinecone_client()
    upserted = 0
    failed: List[str] = []
    batch: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for rec in recordings:
        rec_id = str(rec.get('id') or rec.get('recording_id') or rec.get('file_id') or "").strip()
        if not rec_id:
            log('WARNING', "Skipping record with missing recording_id")
            failed.append("(missing id)")
            continue
        try:
            text = transcripts_service.get_transcript_text(rec_id)
            if not text:
                failed.append(rec_id)
                continue

            # Change-detection: compare hash to existing vector metadata; skip if unchanged
            text_hash = compute_text_hash(text)

            # Try fetch_by_metadata first (more efficient), fallback to fetch by id
            existing_hash = None
            try:
                existing_result = client.fetch_by_metadata(
                    {"recording_id": {"$eq": rec_id}},
                    namespace=target_ns,
                    limit=1,
                )
                vectors = existing_result.get("vectors", {})
                if vectors:
                    first_vec = next(iter(vectors.values()), None)
                    if first_vec and first_vec.get("metadata"):
                        existing_hash = first_vec["metadata"].get("text_hash")
            except Exception:
                # Fallback to fetch by id
                existing = client.fetch_vectors([rec_id], namespace=target_ns).get(rec_id)
                if existing and existing.metadata:
                    existing_hash = existing.metadata.get("text_hash") or existing.metadata.get("hash")

            if existing_hash and existing_hash == text_hash:
                skipped.append(rec_id)
                continue

            embedding = embedder.embed_text(text, dimension=embedder.dimension)
            provider_name = getattr(embedder, 'provider', None)

            # Build validated metadata using schema
            meta = build_metadata(
                recording_id=rec_id,
                text=text,
                model=getattr(embedder, 'model', 'unknown'),
                dimension=embedder.dimension,
                source="plaud",
                provider=provider_name.value if provider_name else None,
                title=rec.get('display_name') or rec.get('name') or rec.get('file_name') or "Untitled",
                start_at=rec.get('start_at') or rec.get('created_at'),
                duration_ms=rec.get('duration') or rec.get('duration_ms'),
                themes=rec.get('themes') or rec.get('tags'),
            )

            batch.append({
                "id": rec_id,
                "values": embedding,
                "metadata": meta,
            })

            if len(batch) >= 50:
                upserted += client.upsert_vectors(batch, namespace=target_ns)
                batch.clear()
        except Exception as e:
            log('ERROR', f"Re-embed failed for {rec_id}: {e}")
            failed.append(rec_id)

    if batch:
        upserted += client.upsert_vectors(batch, namespace=target_ns)

    result = {
        "index": index_name,
        "namespace": target_ns,
        "upserted": upserted,
        "failed": len(failed),
        "skipped": len(skipped),
        "total": total,
    }
    state.pinecone_stats = state.pinecone_stats or {}
    state.pinecone_stats.update({"last_reembed": result})
    log('INFO', f"Re-embedded {upserted}/{total} recordings into {index_name}/{target_ns}; skipped unchanged: {len(skipped)}; failed: {len(failed)}")
    return result


def _persist_index_choice(index_name: str) -> None:
    """Persist chosen index to state and .env for future sessions."""
    state.pinecone_selected_index = index_name
    try:
        settings_service.save_settings({"PINECONE_INDEX_NAME": index_name})
    except Exception as e:
        log('WARNING', f"Could not persist index selection: {e}")


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
