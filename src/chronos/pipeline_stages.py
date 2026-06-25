"""Shared pipeline stage constants and alias normalization."""

CANONICAL_PIPELINE_STAGES: tuple[str, ...] = (
    "full",
    "backfill",
    "ingest",
    "process",
    "index",
    "graph",
    "reindex",
)

PIPELINE_STAGE_ALIASES: dict[str, str] = {
    "all_history": "backfill",
    "full_history": "backfill",
}

ACCEPTED_PIPELINE_STAGES: tuple[str, ...] = CANONICAL_PIPELINE_STAGES + tuple(
    PIPELINE_STAGE_ALIASES.keys()
)


def invalid_pipeline_stage_message(stage: object) -> str:
    return f"Invalid stage: {stage}. Must be one of {list(ACCEPTED_PIPELINE_STAGES)}"


def normalize_pipeline_stage(stage: str) -> str:
    if not isinstance(stage, str):
        raise ValueError(invalid_pipeline_stage_message(stage))

    normalized = stage.strip().lower()
    if not normalized:
        raise ValueError(invalid_pipeline_stage_message(stage))

    canonical = PIPELINE_STAGE_ALIASES.get(normalized, normalized)
    if canonical not in CANONICAL_PIPELINE_STAGES:
        raise ValueError(invalid_pipeline_stage_message(stage))

    return canonical
