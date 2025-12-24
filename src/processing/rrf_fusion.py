"""Reciprocal Rank Fusion (RRF).

RRF is a simple way to merge ranked lists from multiple retrieval sources.
The test suite uses a very small subset of functionality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping


@dataclass(frozen=True)
class RRFResultItem:
    id: str
    score: float
    payload: Dict[str, Any] | None = None


@dataclass(frozen=True)
class RRFMergeResult:
    results: List[RRFResultItem]


def reciprocal_rank_fusion(
    dense_results: List[Mapping[str, Any]],
    sparse_results: List[Mapping[str, Any]],
    keyword_results: List[Mapping[str, Any]],
    *,
    k: int = 60,
) -> RRFMergeResult:
    """Merge 0-3 ranked lists into a single ranked list.

    Inputs are lists of dict-like objects with at least an 'id' key.
    """

    def accumulate(
        items: List[Mapping[str, Any]], scores: MutableMapping[str, float]
    ) -> None:
        for rank, item in enumerate(items, start=1):
            item_id = str(item.get("id"))
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)

    scores: Dict[str, float] = {}
    payloads: Dict[str, Dict[str, Any]] = {}

    for lst in (dense_results, sparse_results, keyword_results):
        for item in lst:
            iid = str(item.get("id"))
            payloads.setdefault(iid, dict(item))

    accumulate(dense_results, scores)
    accumulate(sparse_results, scores)
    accumulate(keyword_results, scores)

    merged = [
        RRFResultItem(id=iid, score=score, payload=payloads.get(iid))
        for iid, score in scores.items()
    ]
    merged.sort(key=lambda r: r.score, reverse=True)

    return RRFMergeResult(results=merged)
