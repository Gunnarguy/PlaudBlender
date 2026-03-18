"""API cost tracker — records every Gemini & OpenAI call with token counts and estimated cost.

Usage:
    from src.chronos.cost_tracker import track_usage, get_cost_summary, get_session_cost

    # After any API call:
    track_usage("gemini-3-flash-preview", "generate", input_tokens=500, output_tokens=200)

    # Get aggregates:
    summary = get_cost_summary()      # all-time by model
    session = get_session_cost()       # since app start
"""

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import logging

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Pricing Table — USD per 1M tokens (as of March 2026)
# Source: https://ai.google.dev/pricing / https://platform.openai.com/docs/pricing
# ═══════════════════════════════════════════════════════════════════

PRICING: dict[str, dict] = {
    # ── Gemini Generation ──────────────────────────────────────────
    "gemini-3-flash-preview": {
        "provider": "google",
        "input_per_mtok": 0.00,  # Free tier
        "output_per_mtok": 0.00,  # Free tier
        "tier": "free",
        "label": "Gemini 3 Flash (free)",
    },
    "gemini-3.1-pro-preview": {
        "provider": "google",
        "input_per_mtok": 1.25,
        "output_per_mtok": 10.00,
        "tier": "paid",
        "label": "Gemini 3.1 Pro",
    },
    "gemini-2.5-pro": {
        "provider": "google",
        "input_per_mtok": 1.25,
        "output_per_mtok": 10.00,
        "tier": "paid",
        "label": "Gemini 2.5 Pro",
    },
    "gemini-2.5-flash": {
        "provider": "google",
        "input_per_mtok": 0.00,
        "output_per_mtok": 0.00,
        "tier": "free",
        "label": "Gemini 2.5 Flash (free)",
    },
    # ── Gemini Embedding ───────────────────────────────────────────
    "gemini-embedding-2-preview": {
        "provider": "google",
        "input_per_mtok": 0.00,
        "output_per_mtok": 0.00,
        "tier": "free",
        "label": "Embedding 2 (free)",
    },
    "gemini-embedding-001": {
        "provider": "google",
        "input_per_mtok": 0.00,
        "output_per_mtok": 0.00,
        "tier": "free",
        "label": "Embedding 001 (free)",
    },
    # ── OpenAI ─────────────────────────────────────────────────────
    "gpt-5.4": {
        "provider": "openai",
        "input_per_mtok": 2.50,
        "output_per_mtok": 15.00,
        "tier": "paid",
        "label": "GPT-5.4",
    },
    "gpt-5.4-pro": {
        "provider": "openai",
        "input_per_mtok": 5.00,
        "output_per_mtok": 30.00,
        "tier": "paid",
        "label": "GPT-5.4 Pro",
    },
    "gpt-5-mini": {
        "provider": "openai",
        "input_per_mtok": 0.25,
        "output_per_mtok": 2.00,
        "tier": "paid",
        "label": "GPT-5 Mini",
    },
    "gpt-5-nano": {
        "provider": "openai",
        "input_per_mtok": 0.10,
        "output_per_mtok": 0.40,
        "tier": "paid",
        "label": "GPT-5 Nano",
    },
    "gpt-5": {
        "provider": "openai",
        "input_per_mtok": 2.00,
        "output_per_mtok": 10.00,
        "tier": "paid",
        "label": "GPT-5",
    },
    "gpt-4.1": {
        "provider": "openai",
        "input_per_mtok": 2.00,
        "output_per_mtok": 8.00,
        "tier": "paid",
        "label": "GPT-4.1",
    },
}


def get_pricing(model: str) -> dict:
    """Return pricing dict for a model, with sensible fallback."""
    if model in PRICING:
        return PRICING[model]
    # Fuzzy match — strip trailing version suffixes
    for key in PRICING:
        if model.startswith(key) or key.startswith(model):
            return PRICING[key]
    # Unknown model — assume paid at mid-range
    return {
        "provider": "unknown",
        "input_per_mtok": 1.00,
        "output_per_mtok": 5.00,
        "tier": "unknown",
        "label": model,
    }


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD for a single API call."""
    p = get_pricing(model)
    cost = (input_tokens / 1_000_000) * p["input_per_mtok"] + (
        output_tokens / 1_000_000
    ) * p["output_per_mtok"]
    return cost


# ═══════════════════════════════════════════════════════════════════
# In-memory session ledger (fast, no I/O)
# ═══════════════════════════════════════════════════════════════════


@dataclass
class UsageRecord:
    model: str
    call_type: str  # "generate", "embed", "search"
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: float  # time.time()


@dataclass
class _SessionLedger:
    records: list[UsageRecord] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    start_time: float = field(default_factory=time.time)

    def add(self, rec: UsageRecord) -> None:
        with self.lock:
            self.records.append(rec)

    def total_cost(self) -> float:
        with self.lock:
            return sum(r.cost_usd for r in self.records)

    def total_tokens(self) -> tuple[int, int]:
        with self.lock:
            inp = sum(r.input_tokens for r in self.records)
            out = sum(r.output_tokens for r in self.records)
            return inp, out

    def by_model(self) -> dict[str, dict]:
        """Aggregate by model."""
        with self.lock:
            agg: dict[str, dict] = {}
            for r in self.records:
                if r.model not in agg:
                    agg[r.model] = {
                        "calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 0.0,
                    }
                agg[r.model]["calls"] += 1
                agg[r.model]["input_tokens"] += r.input_tokens
                agg[r.model]["output_tokens"] += r.output_tokens
                agg[r.model]["cost_usd"] += r.cost_usd
            return agg

    def by_type(self) -> dict[str, dict]:
        """Aggregate by call_type."""
        with self.lock:
            agg: dict[str, dict] = {}
            for r in self.records:
                if r.call_type not in agg:
                    agg[r.call_type] = {
                        "calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 0.0,
                    }
                agg[r.call_type]["calls"] += 1
                agg[r.call_type]["input_tokens"] += r.input_tokens
                agg[r.call_type]["output_tokens"] += r.output_tokens
                agg[r.call_type]["cost_usd"] += r.cost_usd
            return agg

    def recent(self, n: int = 20) -> list[dict]:
        """Most recent n records as dicts."""
        with self.lock:
            return [
                {
                    "model": r.model,
                    "type": r.call_type,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "cost_usd": round(r.cost_usd, 6),
                    "ago_s": round(time.time() - r.timestamp, 1),
                }
                for r in reversed(self.records[-n:])
            ]


_ledger = _SessionLedger()


# ═══════════════════════════════════════════════════════════════════
# SQLite persistence (append-only log for historical cost analysis)
# ═══════════════════════════════════════════════════════════════════

_db_initialized = False
_db_lock = threading.Lock()


def _ensure_table() -> None:
    """Create api_usage_log table if it doesn't exist."""
    global _db_initialized
    if _db_initialized:
        return
    with _db_lock:
        if _db_initialized:
            return
        try:
            from src.database.engine import engine
            from sqlalchemy import text

            with engine.connect() as conn:
                conn.execute(
                    text(
                        """
                    CREATE TABLE IF NOT EXISTS api_usage_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        model TEXT NOT NULL,
                        call_type TEXT NOT NULL,
                        input_tokens INTEGER NOT NULL DEFAULT 0,
                        output_tokens INTEGER NOT NULL DEFAULT 0,
                        cost_usd REAL NOT NULL DEFAULT 0.0,
                        recording_id TEXT,
                        extra TEXT
                    )
                """
                    )
                )
                conn.execute(
                    text(
                        """
                    CREATE INDEX IF NOT EXISTS idx_usage_timestamp
                    ON api_usage_log(timestamp)
                """
                    )
                )
                conn.execute(
                    text(
                        """
                    CREATE INDEX IF NOT EXISTS idx_usage_model
                    ON api_usage_log(model)
                """
                    )
                )
                conn.commit()
            _db_initialized = True
        except Exception as e:
            logger.warning(f"Cost tracker DB init failed (non-fatal): {e}")
            _db_initialized = True  # Don't retry


def _persist(rec: UsageRecord, recording_id: Optional[str] = None) -> None:
    """Write a usage record to SQLite (fire-and-forget)."""
    try:
        _ensure_table()
        from src.database.engine import engine
        from sqlalchemy import text

        ts = datetime.fromtimestamp(rec.timestamp, tz=timezone.utc).isoformat()
        with engine.connect() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO api_usage_log
                        (timestamp, model, call_type, input_tokens, output_tokens, cost_usd, recording_id)
                    VALUES (:ts, :model, :call_type, :inp, :out, :cost, :rid)
                """
                ),
                {
                    "ts": ts,
                    "model": rec.model,
                    "call_type": rec.call_type,
                    "inp": rec.input_tokens,
                    "out": rec.output_tokens,
                    "cost": rec.cost_usd,
                    "rid": recording_id,
                },
            )
            conn.commit()
    except Exception as e:
        logger.debug(f"Cost persist failed (non-fatal): {e}")


# ═══════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════


def track_usage(
    model: str,
    call_type: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    recording_id: Optional[str] = None,
) -> float:
    """Record an API call. Returns estimated cost in USD.

    Args:
        model: Model identifier (e.g. "gemini-3-flash-preview")
        call_type: "generate", "embed", "search", "graph", "community"
        input_tokens: Prompt/input token count
        output_tokens: Completion/output token count
        recording_id: Optional recording being processed

    Returns:
        Estimated cost in USD for this call.
    """
    cost = estimate_cost(model, input_tokens, output_tokens)
    rec = UsageRecord(
        model=model,
        call_type=call_type,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
        timestamp=time.time(),
    )
    _ledger.add(rec)

    # Persist in background to avoid blocking the caller
    t = threading.Thread(target=_persist, args=(rec, recording_id), daemon=True)
    t.start()

    # X-ray telemetry
    try:
        from app_v2.services.xray import xray_log

        cost_str = f"${cost:.4f}" if cost > 0 else "free"
        tok_str = f"{input_tokens:,}→{output_tokens:,} tokens"
        xray_log("data", "cost", f"{model} {call_type}: {tok_str} ({cost_str})")
    except Exception:
        pass

    return cost


def get_session_cost() -> dict:
    """Get session-level cost summary (since app start)."""
    inp, out = _ledger.total_tokens()
    return {
        "total_cost_usd": round(_ledger.total_cost(), 4),
        "total_input_tokens": inp,
        "total_output_tokens": out,
        "total_calls": len(_ledger.records),
        "by_model": _ledger.by_model(),
        "by_type": _ledger.by_type(),
        "recent": _ledger.recent(20),
        "session_start": _ledger.start_time,
        "session_minutes": round((time.time() - _ledger.start_time) / 60, 1),
    }


def get_cost_summary(days: int = 30) -> dict:
    """Get historical cost summary from SQLite.

    Args:
        days: Number of days to look back (default 30).

    Returns:
        Dict with total_cost, by_model, by_day breakdowns.
    """
    try:
        _ensure_table()
        from src.database.engine import engine
        from sqlalchemy import text

        cutoff = datetime.now(tz=timezone.utc)
        cutoff_str = cutoff.replace(
            day=max(1, cutoff.day),
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        ).isoformat()

        with engine.connect() as conn:
            # Total by model
            rows = conn.execute(
                text(
                    """
                SELECT model,
                       COUNT(*) as calls,
                       SUM(input_tokens) as inp,
                       SUM(output_tokens) as out,
                       SUM(cost_usd) as cost
                FROM api_usage_log
                WHERE timestamp >= date('now', :offset)
                GROUP BY model
                ORDER BY cost DESC
            """
                ),
                {"offset": f"-{days} days"},
            ).fetchall()

            by_model = {}
            total_cost = 0.0
            total_calls = 0
            total_inp = 0
            total_out = 0
            for r in rows:
                by_model[r[0]] = {
                    "calls": r[1],
                    "input_tokens": r[2] or 0,
                    "output_tokens": r[3] or 0,
                    "cost_usd": round(r[4] or 0, 4),
                }
                total_cost += r[4] or 0
                total_calls += r[1]
                total_inp += r[2] or 0
                total_out += r[3] or 0

            # By day
            day_rows = conn.execute(
                text(
                    """
                SELECT date(timestamp) as day,
                       SUM(cost_usd) as cost,
                       COUNT(*) as calls,
                       SUM(input_tokens) as inp,
                       SUM(output_tokens) as out
                FROM api_usage_log
                WHERE timestamp >= date('now', :offset)
                GROUP BY date(timestamp)
                ORDER BY day DESC
            """
                ),
                {"offset": f"-{days} days"},
            ).fetchall()

            by_day = [
                {
                    "date": r[0],
                    "cost_usd": round(r[1] or 0, 4),
                    "calls": r[2],
                    "input_tokens": r[3] or 0,
                    "output_tokens": r[4] or 0,
                }
                for r in day_rows
            ]

        return {
            "days": days,
            "total_cost_usd": round(total_cost, 4),
            "total_calls": total_calls,
            "total_input_tokens": total_inp,
            "total_output_tokens": total_out,
            "by_model": by_model,
            "by_day": by_day,
        }

    except Exception as e:
        logger.warning(f"Cost summary query failed: {e}")
        return {
            "days": days,
            "total_cost_usd": 0,
            "total_calls": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "by_model": {},
            "by_day": [],
        }


def get_model_pricing_table() -> list[dict]:
    """Return pricing info for all known models, sorted by provider then cost."""
    rows = []
    for model_id, info in sorted(
        PRICING.items(), key=lambda x: (x[1]["provider"], x[1]["input_per_mtok"])
    ):
        rows.append(
            {
                "model": model_id,
                "label": info["label"],
                "provider": info["provider"],
                "input_per_mtok": info["input_per_mtok"],
                "output_per_mtok": info["output_per_mtok"],
                "tier": info["tier"],
            }
        )
    return rows
