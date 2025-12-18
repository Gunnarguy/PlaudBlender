from __future__ import annotations

"""
Vector DB service wrapper (provider-neutral).

All operations flow through the vector store abstraction (`VECTOR_DB`, Qdrant by
default). This is the canonical service to use from the GUI and other modules.
"""

from typing import List, Dict, Tuple, Any, Optional

import os
from gui.services.clients import get_vector_db_client
from gui.services.embedding_service import get_embedding_service
from gui.services import transcripts_service, settings_service
from gui.state import state
from gui.utils.logging import log


def get_collections_and_namespaces() -> (
    Tuple[List[str], List[str], str, Dict[str, Any]]
):
    """Return (collections, namespaces, current_collection, stats_dict)."""
    embedder = get_embedding_service()
    target_dim = embedder.dimension
    provider_name = getattr(embedder, "provider", None)
    client = get_vector_db_client()
    collections = client.list_indexes()
    try:
        from src.vector_store import is_qdrant

        db_name = "Qdrant" if is_qdrant() else "Vector Store"
        noun = "collections" if is_qdrant() else "indexes"
    except Exception:
        db_name = "Vector Store"
        noun = "indexes"
    log("INFO", f"{db_name} {noun}: {collections}")
    raw_namespaces = client.list_namespaces()
    current = client.index_name

    info = client.get_index_info()
    ns_from_info = (
        list(info.get("namespaces", {}).keys()) if isinstance(info, dict) else []
    )
    namespaces = ns_from_info or [ns for ns in raw_namespaces if ns is not None]

    stats = {
        "vectors": info.get("total_vectors", 0) if isinstance(info, dict) else 0,
        "dimension": info.get("dimension", "—") if isinstance(info, dict) else "—",
        "metric": info.get("metric", "—") if isinstance(info, dict) else "—",
        "namespaces": len(namespaces),
        "namespace_names": namespaces,
        "target_dim": target_dim,
        "dim_mismatch": (
            (info.get("dimension") != target_dim) if isinstance(info, dict) else False
        ),
        "provider": provider_name.value if provider_name else "google",
        "index_name": info.get("name") if isinstance(info, dict) else client.index_name,
    }

    state.vector_collections = collections
    state.pinecone_indexes = collections  # legacy alias
    state.vector_stats = stats
    state.pinecone_stats = stats  # legacy alias
    log(
        "INFO",
        f"Found {len(collections)} collections, {len(namespaces)} namespaces, {stats['vectors']} vectors; "
        f"current collection={stats.get('index_name')}, dim={stats.get('dimension')}, target_dim={target_dim}",
    )
    return collections, namespaces, current, stats


def switch_collection(collection_name: str) -> Tuple[List[str], Dict[str, Any]]:
    """Switch collection/index and return (namespaces, stats)."""
    client = get_vector_db_client()
    log("INFO", f"Switching active collection -> {collection_name}")
    client.switch_index(collection_name)
    _persist_collection_choice(collection_name)
    namespaces = client.list_namespaces()
    info = client.get_index_info()
    embedder = get_embedding_service()
    target_dim = embedder.dimension
    provider_name = getattr(embedder, "provider", None)
    stats = {
        "vectors": info.get("total_vectors", 0),
        "dimension": info.get("dimension", "—"),
        "metric": info.get("metric", "—"),
        "namespaces": len(namespaces),
        "index_name": info.get("name", collection_name),
        "target_dim": target_dim,
        "dim_mismatch": (info.get("dimension") != target_dim),
        "provider": provider_name.value if provider_name else "google",
    }
    log(
        "INFO",
        f"Switched to {stats['index_name']} (dim={stats['dimension']}, target={target_dim}, "
        f"namespaces={len(namespaces)}, vectors={stats['vectors']})",
    )
    return namespaces, stats


def refresh_vectors(namespace: str = "") -> List[Dict]:
    client = get_vector_db_client()
    if not namespace:
        fallback_ns = (
            state.vector_stats.get("namespace_names") if state.vector_stats else None
        )
        namespace = (
            (fallback_ns or [""])[0]
            if isinstance(fallback_ns, list) and fallback_ns
            else ""
        )
    vectors = client.get_all_vectors(namespace)
    formatted = [_format_vector(vec) for vec in vectors]
    formatted.sort(key=lambda v: v.get("date", ""), reverse=True)
    state.vector_vectors = formatted
    state.pinecone_vectors = formatted  # legacy alias
    try:
        from src.vector_store import is_qdrant

        db_name = "Qdrant" if is_qdrant() else "Vector Store"
    except Exception:
        db_name = "Vector Store"
    log("INFO", f"Loaded {len(formatted)} vectors from {db_name}")
    return formatted


def update_vector_metadata(
    vec_id: str, metadata: Dict[str, Any], namespace: str = ""
) -> bool:
    """Update metadata for a vector and return success."""
    client = get_vector_db_client()
    return client.update_metadata(vec_id, metadata, namespace=namespace)


def delete_vectors(
    ids: Optional[List[str]] = None,
    flt: Optional[Dict] = None,
    delete_all: bool = False,
    namespace: str = "",
) -> str:
    """Delete vectors by ids, filter, or delete_all. Returns a status string."""
    client = get_vector_db_client()
    if delete_all:
        ok = client.delete_all(namespace=namespace)
        return "all" if ok else "error"
    if flt is not None:
        ok = client.delete_by_filter(flt, namespace=namespace)
        return "filtered" if ok else "error"
    ids = ids or []
    ok = client.delete_vectors(ids, namespace=namespace)
    return str(len(ids)) if ok else "error"


def ensure_matching_collection(target_collection: str, dimension: int) -> str:
    """Ensure a collection exists with the given name and dimension."""
    client = get_vector_db_client()
    collections = client.list_indexes()
    dims = getattr(client, "list_index_dimensions", lambda: {})()
    log(
        "INFO",
        f"Ensuring matching collection '{target_collection}' at dim {dimension}; existing dims: {dims}",
    )
    if target_collection in dims and dims[target_collection] != dimension:
        raise ValueError(
            f"Collection '{target_collection}' has dimension {dims[target_collection]}, expected {dimension}."
        )
    if target_collection not in collections:
        created = client.create_index(name=target_collection, dimension=dimension)
        if not created:
            raise RuntimeError(f"Failed to create collection {target_collection}")
        log("INFO", f"Created collection '{target_collection}' at dim {dimension}")
    client.switch_index(target_collection)
    _persist_collection_choice(target_collection)
    return target_collection


def find_matching_collection(dimension: int) -> Optional[str]:
    """Return an existing collection name that matches dimension, if any."""
    client = get_vector_db_client()
    dims = getattr(client, "list_index_dimensions", lambda: {})()
    for name, dim in dims.items():
        if dim == dimension:
            return name
    return None


def reembed_all_into_collection(collection_name: str, namespace: str = "") -> None:
    """Re-embed all Plaud recordings into the target collection/namespace."""
    from src.models.vector_metadata import build_metadata, compute_text_hash

    log(
        "INFO",
        f"Re-embed start -> collection={collection_name}, namespace={namespace or 'full_text'}",
    )
    embedder = get_embedding_service()
    target_ns = namespace or "full_text"
    ensure_matching_collection(collection_name, embedder.dimension)

    recordings = transcripts_service.fetch_transcripts()
    total = len(recordings)
    log("INFO", f"Re-embed fetched {total} recordings")
    if not recordings:
        return {
            "collection": collection_name,
            "namespace": target_ns,
            "upserted": 0,
            "failed": 0,
            "total": 0,
        }

    client = get_vector_db_client()
    upserted = 0
    failed: List[str] = []
    batch: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for rec in recordings:
        rec_id = str(
            rec.get("id") or rec.get("recording_id") or rec.get("file_id") or ""
        ).strip()
        if not rec_id:
            log("WARNING", "Skipping record with missing recording_id")
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

                # Normalize shapes across providers (Qdrant returns list[dict],
                # Legacy compatibility layer returns .vectors or {"vectors": {...}})
                vectors = {}
                if isinstance(existing_result, dict):
                    vectors = existing_result.get("vectors", {}) or {}
                elif hasattr(existing_result, "vectors"):
                    vectors = getattr(existing_result, "vectors", {}) or {}
                elif isinstance(existing_result, list):
                    vectors = {
                        (v.get("metadata", {}) or {}).get("_original_id")
                        or str(v.get("id"))
                        or rec_id: v
                        for v in existing_result
                    }

                if vectors:
                    first_vec = next(iter(vectors.values()), None)
                    meta = None
                    if hasattr(first_vec, "metadata"):
                        meta = first_vec.metadata
                    elif isinstance(first_vec, dict):
                        meta = first_vec.get("metadata")
                    if meta:
                        existing_hash = meta.get("text_hash")
                        if not existing_hash:
                            existing_hash = meta.get("hash")
            except Exception:
                # Fallback to fetch by id (also provider-agnostic shape)
                existing = client.fetch_vectors([rec_id], namespace=target_ns).get(
                    rec_id
                )
                meta = None
                if hasattr(existing, "metadata"):
                    meta = existing.metadata
                elif isinstance(existing, dict):
                    meta = existing.get("metadata")
                if meta:
                    existing_hash = meta.get("text_hash") or meta.get("hash")

            if existing_hash and existing_hash == text_hash:
                skipped.append(rec_id)
                continue

            embedding = embedder.embed_text(text, dimension=embedder.dimension)
            provider_name = getattr(embedder, "provider", None)

            # Build validated metadata using schema
            meta = build_metadata(
                recording_id=rec_id,
                text=text,
                model=getattr(embedder, "model", "unknown"),
                dimension=embedder.dimension,
                source="plaud",
                provider=provider_name.value if provider_name else None,
                title=rec.get("display_name")
                or rec.get("name")
                or rec.get("file_name")
                or "Untitled",
                start_at=rec.get("start_at") or rec.get("created_at"),
                duration_ms=rec.get("duration") or rec.get("duration_ms"),
                themes=rec.get("themes") or rec.get("tags"),
            )

            batch.append(
                {
                    "id": rec_id,
                    "values": embedding,
                    "metadata": meta,
                }
            )

            if len(batch) >= 50:
                upserted += client.upsert_vectors(batch, namespace=target_ns)
                batch.clear()
        except Exception as e:
            log("ERROR", f"Re-embed failed for {rec_id}: {e}")
            failed.append(rec_id)

    if batch:
        upserted += client.upsert_vectors(batch, namespace=target_ns)

    result = {
        "collection": collection_name,
        "namespace": target_ns,
        "upserted": upserted,
        "failed": len(failed),
        "skipped": len(skipped),
        "total": total,
    }
    state.vector_stats = state.vector_stats or {}
    state.vector_stats.update({"last_reembed": result})
    state.pinecone_stats = state.vector_stats  # legacy alias
    log(
        "INFO",
        f"Re-embedded {upserted}/{total} recordings into {collection_name}/{target_ns}; skipped unchanged: {len(skipped)}; failed: {len(failed)}",
    )
    return result


def _persist_collection_choice(collection_name: str) -> None:
    """Persist chosen collection to state and .env for future sessions."""
    state.vector_selected_collection = collection_name
    state.pinecone_selected_index = collection_name  # legacy alias
    try:
        # Preserve backward compatibility by writing both keys if present
        settings_service.save_settings(
            {
                "QDRANT_COLLECTION": collection_name,
                "PINECONE_INDEX_NAME": collection_name,
            }
        )
    except Exception as e:
        log("WARNING", f"Could not persist collection selection: {e}")


def _format_vector(vec) -> Dict:
    """Normalize vector shape across providers and format for UI."""
    meta = None
    if hasattr(vec, "metadata"):
        meta = vec.metadata
    elif isinstance(vec, dict):
        meta = vec.get("metadata")
    meta = meta or {}

    vec_id = (
        getattr(vec, "id", None)
        or (vec.get("id") if isinstance(vec, dict) else None)
        or ""
    )
    title = meta.get("title") or meta.get("name") or "Untitled"
    duration_ms = meta.get("duration_ms") or meta.get("duration") or 0
    minutes = int(duration_ms // 60000)
    seconds = int((duration_ms % 60000) // 1000)
    duration = f"{minutes}:{seconds:02d}" if duration_ms else "—"

    return {
        "id": vec_id,
        "short_id": f"{vec_id[:10]}…" if vec_id else "—",
        "title": title,
        "date": meta.get("date") or meta.get("start_at", "")[:10],
        "duration": duration,
        "tags": meta.get("themes") or meta.get("tags") or "—",
        "field_count": len(meta),
        "metadata": meta,
    }
