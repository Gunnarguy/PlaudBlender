"""Components package."""

import os

# ── Canonical category palette (single source of truth) ───────────────────────
CATEGORIES = [
    "work",
    "personal",
    "meeting",
    "reflection",
    "idea",
    "deep_work",
    "break",
    "unknown",
]

CATEGORY_COLORS = {
    "work": "#3b82f6",  # blue
    "personal": "#8b5cf6",  # purple
    "meeting": "#f59e0b",  # amber
    "reflection": "#10b981",  # emerald
    "idea": "#ec4899",  # pink
    "deep_work": "#6366f1",  # indigo
    "break": "#64748b",  # slate
    "unknown": "#374151",  # gray
}

# Extend with user-defined custom categories from env
_custom_raw = os.environ.get("CHRONOS_CUSTOM_CATEGORIES", "")
_CUSTOM_PALETTE = ["#f97316", "#06b6d4", "#84cc16", "#e11d48", "#14b8a6", "#a855f7"]
for _i, _cat in enumerate(c.strip().lower().replace(" ", "_") for c in _custom_raw.split(",") if c.strip()):
    if _cat not in CATEGORIES:
        CATEGORIES.append(_cat)
        CATEGORY_COLORS[_cat] = _CUSTOM_PALETTE[_i % len(_CUSTOM_PALETTE)]

CATEGORY_LABELS = {k: k.replace("_", " ").title() for k in CATEGORIES}

from app_v2.components.sidebar import create_sidebar
from app_v2.components.day_view import (
    create_day_view,
    create_day_card,
    create_recording_card,
)
from app_v2.components.recording_detail import (
    create_recording_detail,
    create_recording_placeholder,
)
from app_v2.components.search import create_search_bar, create_search_results
from app_v2.components.topics import create_topics_grid, create_topic_timeline_view
from app_v2.components.stats import create_stats_view
from app_v2.components.graph import create_graph_view

__all__ = [
    "CATEGORIES",
    "CATEGORY_COLORS",
    "CATEGORY_LABELS",
    "create_sidebar",
    "create_day_view",
    "create_day_card",
    "create_recording_card",
    "create_recording_detail",
    "create_recording_placeholder",
    "create_search_bar",
    "create_search_results",
    "create_topics_grid",
    "create_topic_timeline_view",
    "create_stats_view",
    "create_graph_view",
]
