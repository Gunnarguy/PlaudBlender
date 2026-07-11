"""
Pipeline Progress Tracker — shared-state progress file for live UI updates.

The pipeline writes JSON progress to ``data/pipeline_progress.json``
every time a phase/step changes.  The Dash UI reads the same file on a
short interval to render a live progress panel.

Usage (writer side — in pipeline code):
    from src.chronos.pipeline_progress import progress

    progress.start_run(phases=["ingest", "process", "index"])
    progress.start_phase("ingest", total_items=12)
    progress.update(step="Fetching recording list from Plaud…")
    progress.advance(item="rec_abc123")      # increments completed_items
    progress.finish_phase(summary="Ingested 12 recordings")
    progress.finish_run()

Usage (reader side — in Dash callback):
    from src.chronos.pipeline_progress import read_progress
    data = read_progress()   # dict or None
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

PROGRESS_FILE = Path("data/pipeline_progress.json")


@dataclass
class PhaseProgress:
    name: str
    status: str = "pending"  # pending | running | completed | failed
    total_items: int = 0
    completed_items: int = 0
    current_step: str = ""
    current_item: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    elapsed_seconds: float = 0.0
    summary: str = ""
    error: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class PipelineRun:
    run_id: str = ""
    status: str = "idle"  # idle | running | completed | failed
    phases: list[PhaseProgress] = field(default_factory=list)
    current_phase: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    elapsed_seconds: float = 0.0
    trigger: str = ""  # manual | scheduled | webhook | usb
    sync_mode: str = ""
    partial_success: bool = False
    warning: str = ""
    warnings: list[str] = field(default_factory=list)


class PipelineProgressTracker:
    """Write-side tracker. Call methods as the pipeline progresses."""

    def __init__(self) -> None:
        self._run = PipelineRun()
        self._phase_map: dict[str, PhaseProgress] = {}

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(
        self,
        phases: list[str],
        trigger: str = "manual",
    ) -> str:
        """Begin a new pipeline run. Returns run_id."""
        run_id = uuid.uuid4().hex[:12]
        self._run = PipelineRun(
            run_id=run_id,
            status="running",
            started_at=time.time(),
            trigger=trigger,
            phases=[PhaseProgress(name=p) for p in phases],
        )
        self._phase_map = {p.name: p for p in self._run.phases}
        self._flush()
        return run_id

    def finish_run(self, error: str = "") -> None:
        finished_at = time.time()
        final_status = "failed" if error else "completed"

        for p in self._run.phases:
            if p.status != "running":
                continue
            p.status = final_status
            if not p.started_at:
                p.started_at = self._run.started_at or finished_at
            p.finished_at = finished_at
            p.elapsed_seconds = finished_at - p.started_at
            if error and not p.error:
                p.error = error

        self._run.status = final_status
        self._run.finished_at = finished_at
        self._run.elapsed_seconds = self._run.finished_at - self._run.started_at
        self._flush()

    # ------------------------------------------------------------------
    # Phase lifecycle
    # ------------------------------------------------------------------

    def start_phase(self, phase: str, total_items: int = 0) -> None:
        p = self._phase_map.get(phase)
        if not p:
            p = PhaseProgress(name=phase)
            self._run.phases.append(p)
            self._phase_map[phase] = p
        p.status = "running"
        p.total_items = total_items
        p.completed_items = 0
        p.started_at = time.time()
        self._run.current_phase = phase
        self._flush()

    def finish_phase(self, phase: str | None = None, summary: str = "", error: str = "") -> None:
        phase = phase or self._run.current_phase
        p = self._phase_map.get(phase)
        if not p:
            return
        p.status = "failed" if error else "completed"
        p.finished_at = time.time()
        p.elapsed_seconds = p.finished_at - p.started_at
        p.summary = summary
        p.error = error
        self._flush()

    def set_phase_warnings(
        self,
        phase: str | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        phase = phase or self._run.current_phase
        p = self._phase_map.get(phase)
        if not p:
            return
        p.warnings = _normalize_messages(warnings)
        self._flush()

    def set_run_context(
        self,
        *,
        sync_mode: str | None = None,
        partial_success: bool | None = None,
        warning: str | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        if sync_mode is not None:
            self._run.sync_mode = sync_mode
        if partial_success is not None:
            self._run.partial_success = partial_success
        if warning is not None:
            self._run.warning = warning.strip()
        if warnings is not None:
            normalized = _normalize_messages(warnings)
            self._run.warnings = normalized
            if normalized and not self._run.warning:
                self._run.warning = normalized[0]
            elif not normalized and warning is None:
                self._run.warning = ""
        self._flush()

    # ------------------------------------------------------------------
    # Step-level updates (within a phase)
    # ------------------------------------------------------------------

    def update(
        self,
        step: str = "",
        item: str = "",
        total: int | None = None,
        completed: int | None = None,
    ) -> None:
        """Update current step description, total, and completed count."""
        p = self._phase_map.get(self._run.current_phase)
        if not p:
            return
        if step:
            p.current_step = step
        if item:
            p.current_item = item
        if total is not None:
            p.total_items = total
        if completed is not None:
            p.completed_items = completed
        p.elapsed_seconds = time.time() - p.started_at
        self._run.elapsed_seconds = time.time() - self._run.started_at
        self._flush()


    def advance(self, item: str = "", step: str = "") -> None:
        """Increment completed_items by 1 and optionally set step/item."""
        p = self._phase_map.get(self._run.current_phase)
        if not p:
            return
        p.completed_items += 1
        if item:
            p.current_item = item
        if step:
            p.current_step = step
        p.elapsed_seconds = time.time() - p.started_at
        self._run.elapsed_seconds = time.time() - self._run.started_at
        self._flush()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _flush(self) -> None:
        """Write current state to JSON file (atomic via tmp+rename)."""
        try:
            PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "run_id": self._run.run_id,
                "status": self._run.status,
                "current_phase": self._run.current_phase,
                "started_at": self._run.started_at,
                "finished_at": self._run.finished_at,
                "elapsed_seconds": round(time.time() - self._run.started_at, 1)
                if self._run.started_at
                else 0,
                "trigger": self._run.trigger,
                "sync_mode": self._run.sync_mode or None,
                "partial_success": self._run.partial_success,
                "warning": self._run.warning or None,
                "warnings": list(self._run.warnings),
                "phases": [asdict(p) for p in self._run.phases],
            }
            tmp = PROGRESS_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            tmp.rename(PROGRESS_FILE)
        except Exception:
            logger.debug("Failed to write pipeline progress", exc_info=True)


# Module-level singleton — importable from anywhere
progress = PipelineProgressTracker()


def _normalize_messages(messages: list[str] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()

    for message in messages or []:
        text = str(message or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)

    return normalized


def read_progress() -> dict[str, Any] | None:
    """Read-side: return latest progress dict, or None if no run."""
    try:
        if not PROGRESS_FILE.exists():
            return None
        data = json.loads(PROGRESS_FILE.read_text())
        normalized = False
        if not data.get("sync_mode"):
            current_phase = str(data.get("current_phase", "")).strip().lower()
            phase_names = {
                str(phase.get("name", "")).strip().lower()
                for phase in data.get("phases", [])
            }
            if current_phase == "backfill" or "backfill" in phase_names:
                data["sync_mode"] = "backfill"
                normalized = True
        final_status = str(data.get("status", "")).lower()
        if final_status in {"completed", "failed", "idle"}:
            phase_final_status = "failed" if final_status == "failed" else "completed"
            for phase in data.get("phases", []):
                if phase.get("status") != "running":
                    continue
                normalized = True
                phase["status"] = phase_final_status
                if not phase.get("finished_at"):
                    phase["finished_at"] = data.get("finished_at") or time.time()
                if not phase.get("started_at"):
                    phase["started_at"] = data.get("started_at") or phase["finished_at"]
                phase["elapsed_seconds"] = max(
                    0,
                    float(phase["finished_at"]) - float(phase["started_at"]),
                )
        data.setdefault("partial_success", False)
        data.setdefault("warning", None)
        data.setdefault("warnings", [])
        for phase in data.get("phases", []):
            phase.setdefault("warnings", [])
        if normalized:
            tmp = PROGRESS_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            tmp.rename(PROGRESS_FILE)
        # Add a computed "age" so the UI can decide whether data is stale
        if data.get("started_at"):
            data["age_seconds"] = round(time.time() - data["started_at"], 1)

        # Supplement with real-time cost and active models telemetry
        run_id = data.get("run_id")
        if run_id:
            try:
                from src.chronos.cost_tracker import get_run_cost_details
                details = get_run_cost_details(run_id)
                data["accumulated_cost_usd"] = details.get("total_cost", 0.0)
                data["activated_models"] = details.get("models", [])
            except Exception:
                data["accumulated_cost_usd"] = 0.0
                data["activated_models"] = []
        else:
            data["accumulated_cost_usd"] = 0.0
            data["activated_models"] = []

        return data
    except (json.JSONDecodeError, OSError):
        return None
