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
        self._run.status = "failed" if error else "completed"
        self._run.finished_at = time.time()
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

    # ------------------------------------------------------------------
    # Step-level updates (within a phase)
    # ------------------------------------------------------------------

    def update(self, step: str = "", item: str = "", total: int | None = None) -> None:
        """Update current step description without advancing the counter."""
        p = self._phase_map.get(self._run.current_phase)
        if not p:
            return
        if step:
            p.current_step = step
        if item:
            p.current_item = item
        if total is not None:
            p.total_items = total
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
                "phases": [asdict(p) for p in self._run.phases],
            }
            tmp = PROGRESS_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            tmp.rename(PROGRESS_FILE)
        except Exception:
            logger.debug("Failed to write pipeline progress", exc_info=True)


# Module-level singleton — importable from anywhere
progress = PipelineProgressTracker()


def read_progress() -> dict[str, Any] | None:
    """Read-side: return latest progress dict, or None if no run."""
    try:
        if not PROGRESS_FILE.exists():
            return None
        data = json.loads(PROGRESS_FILE.read_text())
        # Add a computed "age" so the UI can decide whether data is stale
        if data.get("started_at"):
            data["age_seconds"] = round(time.time() - data["started_at"], 1)
        return data
    except (json.JSONDecodeError, OSError):
        return None
