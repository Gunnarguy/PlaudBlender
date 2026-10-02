"""
GraphRAG Entity Extraction Module for PlaudBlender

Extracts entities and relationships from transcripts to build a knowledge graph.
This enables multi-hop reasoning queries like:
- "What projects involve both Alice and Bob?"
- "Which meetings discussed budget AND were attended by the CEO?"

Entity Types:
- PERSON: People mentioned in transcripts
- PROJECT: Projects, initiatives, products
- TOPIC: Key themes and subjects
- ACTION: Action items and tasks
- DATE: Temporal references
- METRIC: Numbers, KPIs, measurements

Reference: gemini-deep-research-RAG.txt Section on GraphRAG
"""

import os
import re
import json
import time
import logging
import hashlib
from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from dotenv import load_dotenv

from src.chronos.genai_helpers import get_genai_client, pick_first_available_or_known
from src.config import get_settings

load_dotenv()
logger = logging.getLogger(__name__)
_NO_ENTITY_WARNING_MIN_TEXT_LEN = 120


@dataclass
class _CompletionResult:
    """Minimal compatibility shim for llama_index-style .complete() results."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class EntityType(str, Enum):
    """Types of entities to extract from transcripts."""

    PERSON = "person"
    PROJECT = "project"
    TOPIC = "topic"
    ACTION = "action"
    DATE = "date"
    METRIC = "metric"
    ORGANIZATION = "organization"
    LOCATION = "location"


class RelationType(str, Enum):
    """Types of relationships between entities."""

    MENTIONS = "mentions"  # Document mentions entity
    RELATED_TO = "related_to"  # Entity is related to another
    ASSIGNED_TO = "assigned_to"  # Action assigned to person
    DISCUSSED_IN = "discussed_in"  # Topic discussed in meeting
    WORKS_ON = "works_on"  # Person works on project
    REPORTS_TO = "reports_to"  # Person reports to another
    DEADLINE = "deadline"  # Action has deadline
    WORKS_WITH = "works_with"  # Person works alongside another person
    MEMBER_OF = "member_of"  # Person belongs to an organization or team
    PART_OF = "part_of"  # Project/topic is part of a larger one
    USES = "uses"  # Person/project uses a tool or technology
    LOCATED_IN = "located_in"  # Something happens at / belongs to a place
    KNOWS = "knows"  # Personal tie (friend, family, acquaintance)
    CO_MENTIONED = "co_mentioned"  # Mentioned in the same moment (derived, no model call)


# Relations the model may state; CO_MENTIONED and MENTIONS are derived, never asked for.
STATED_RELATIONS = {
    "works_with", "works_on", "member_of", "reports_to", "assigned_to",
    "part_of", "uses", "located_in", "knows", "related_to",
}
LINKABLE_TYPES = {"person", "project", "topic", "organization", "location"}
_HONORIFIC = re.compile(r"^(dr|mr|mrs|ms|mx|miss|prof|professor|sir)\.?\s+", re.IGNORECASE)


@dataclass
class Entity:
    """Represents an extracted entity."""

    id: str
    name: str
    entity_type: EntityType
    aliases: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    mention_count: int = 1

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.entity_type.value,
            "aliases": self.aliases,
            "metadata": self.metadata,
            "mention_count": self.mention_count,
        }

    @staticmethod
    def canonical_name(name: str, entity_type: EntityType) -> str:
        """Lower-cased, trimmed name; people also lose an honorific ("Dr. Patel" == "Patel")."""
        text = re.sub(r"\s+", " ", str(name or "")).strip().strip(" .,;:!?\"'()[]")
        if entity_type == EntityType.PERSON:
            text = _HONORIFIC.sub("", text)
        return text.casefold()

    @staticmethod
    def generate_id(name: str, entity_type: EntityType) -> str:
        """Generate deterministic entity ID (from the canonical name)."""
        content = f"{entity_type.value}:{Entity.canonical_name(name, entity_type)}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class Relationship:
    """Represents a relationship between two entities."""

    source_id: str
    target_id: str
    relation_type: RelationType
    weight: float = 1.0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.relation_type.value,
            "weight": self.weight,
            "metadata": self.metadata,
        }


@dataclass
class KnowledgeGraph:
    """In-memory knowledge graph with entities and relationships."""

    entities: Dict[str, Entity] = field(default_factory=dict)
    relationships: List[Relationship] = field(default_factory=list)
    _adjacency: Dict[str, List[Relationship]] = field(
        default_factory=dict, init=False, repr=False
    )
    document_entities: Dict[str, Set[str]] = field(
        default_factory=dict
    )  # doc_id -> entity_ids

    def add_entity(self, entity: Entity) -> None:
        """Add or merge an entity."""
        if entity.id in self.entities:
            # Merge: increment mention count, add aliases
            existing = self.entities[entity.id]
            existing.mention_count += entity.mention_count
            for alias in entity.aliases:
                if alias not in existing.aliases:
                    existing.aliases.append(alias)
        else:
            self.entities[entity.id] = entity

    def add_relationship(self, rel: Relationship) -> None:
        """Add a relationship (allows duplicates, increments weight)."""
        # Check for existing relationship
        for existing in self._adjacency.get(rel.source_id, []):
            if (
                existing.target_id == rel.target_id
                and existing.relation_type == rel.relation_type
            ):
                existing.weight += rel.weight
                for quote in (rel.metadata or {}).get("evidence", []):
                    bucket = existing.metadata.setdefault("evidence", [])
                    if quote not in bucket and len(bucket) < 3:
                        bucket.append(quote)
                return
        self.relationships.append(rel)

        # Add to adjacency list
        if rel.source_id not in self._adjacency:
            self._adjacency[rel.source_id] = []
        self._adjacency[rel.source_id].append(rel)

        if rel.target_id not in self._adjacency:
            self._adjacency[rel.target_id] = []
        if rel.target_id != rel.source_id:  # avoid duplicating self-loops
            self._adjacency[rel.target_id].append(rel)

    def link_document(self, doc_id: str, entity_ids: List[str]) -> None:
        """Link a document to its extracted entities."""
        if doc_id not in self.document_entities:
            self.document_entities[doc_id] = set()
        self.document_entities[doc_id].update(entity_ids)

    def get_related_documents(self, entity_id: str) -> List[str]:
        """Find all documents that mention an entity."""
        docs = []
        for doc_id, entities in self.document_entities.items():
            if entity_id in entities:
                docs.append(doc_id)
        return docs

    def get_entity_neighbors(self, entity_id: str) -> List[Tuple[Entity, Relationship]]:
        """Get all entities connected to a given entity."""
        neighbors = []
        for rel in self._adjacency.get(entity_id, []):
            if rel.source_id == entity_id:
                target = self.entities.get(rel.target_id)
                if target:
                    neighbors.append((target, rel))
            elif rel.target_id == entity_id:
                source = self.entities.get(rel.source_id)
                if source:
                    neighbors.append((source, rel))
        return neighbors

    def search_entities(
        self, query: str, entity_type: Optional[EntityType] = None
    ) -> List[Entity]:
        """Search entities by name or alias."""
        query_lower = query.lower()
        matches = []
        for entity in self.entities.values():
            if entity_type and entity.entity_type != entity_type:
                continue
            if query_lower in entity.name.lower():
                matches.append(entity)
            elif any(query_lower in alias.lower() for alias in entity.aliases):
                matches.append(entity)
        return sorted(matches, key=lambda e: e.mention_count, reverse=True)

    def to_dict(self) -> Dict:
        return {
            "entities": [e.to_dict() for e in self.entities.values()],
            "relationships": [r.to_dict() for r in self.relationships],
            "document_count": len(self.document_entities),
        }

    def stats(self) -> Dict:
        """Get graph statistics."""
        type_counts = {}
        for entity in self.entities.values():
            t = entity.entity_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "documents_indexed": len(self.document_entities),
            "entities_by_type": type_counts,
        }



def _is_reasoning_model(model: str) -> bool:
    """True for OpenAI models that reject an explicit `temperature`."""
    m = (model or "").strip().lower()
    return (
        m.startswith("gpt-5.6")
        or m.startswith("gpt-5.5")
        or m.startswith("o1")
        or m.startswith("o3")
        or m.startswith("o4")
    )


class EntityExtractor:
    """
    Extracts entities from transcript text using LLM.

    Uses structured prompting to identify:
    - People (speakers, mentioned individuals)
    - Projects/Products
    - Topics/Themes
    - Action Items
    - Dates/Deadlines
    - Metrics/Numbers

    Usage:
        extractor = EntityExtractor()
        entities = extractor.extract_entities(
            text="In the Q3 review, Alice mentioned the Alpha project is behind schedule...",
            doc_id="recording_123",
        )
    """

    EXTRACTION_PROMPT = """Extract entities from this transcript text. Return a JSON object with the following structure:

{
  "people": [{"name": "Person Name", "role": "optional role/title"}],
  "projects": [{"name": "Project Name", "status": "optional status"}],
  "topics": ["Concrete Subject Noun 1", "Specific Subject 2"],
  "actions": [{"task": "Task description", "assignee": "Person (if mentioned)", "deadline": "Date (if mentioned)"}],
  "dates": ["2024-10-15", "Q3 2024"],
  "metrics": [{"value": "15%", "context": "revenue growth"}],
  "organizations": ["Company/Org Name"],
  "locations": ["Place Name"],
  "relationships": [{"source": "Name", "source_type": "person", "relation": "works_with", "target": "Name", "target_type": "person", "evidence": "<=15 words from the text"}]
}

Relationship relation values: works_with, works_on, member_of, reports_to, assigned_to, part_of, uses, located_in, knows, related_to.
Types: person, project, topic, organization, location. Only relationships the text states or clearly implies.

CRITICAL TOPIC RULES:
- Topics MUST be concrete subject nouns, proper nouns, technologies, projects, or multi-word concept phrases (e.g., "Raspberry Pi", "Notion Sync", "API Billing", "iOS App", "Tailscale").
- NEVER extract single verbs, action words, gerunds, or conversational filler as topics (e.g. NEVER output "going", "using", "swapping", "talking", "asking", "doing", "running", "wants").
- Every topic must be a distinct, meaningful subject entity.

Only include entities that are clearly mentioned. Be specific with names.
If an entity is not found, use an empty array.

Transcript:
{text}

Return ONLY the JSON object, no other text."""

    def __init__(self, llm=None):
        """
        Initialize entity extractor.

        Args:
            llm: LLM instance (defaults to the configured Chronos provider)
        """
        self.settings = get_settings()
        provider = (
            (
                getattr(self.settings, "chronos_processing_provider", "gemini")
                or "gemini"
            )
            .strip()
            .lower()
        )
        if provider == "agy":
            self._provider = "agy"
        else:
            self._provider = "gemini" if provider == "gemini" else "openai"
        self.min_interval = 0.1
        self._last_call_ts: float = 0.0
        self._agy = None

        if llm is None:
            if self._provider == "agy":
                # Subscription path (AI Ultra via the host's AGY bridge). No metered
                # fallback here: a skipped event only costs the graph one node.
                from src.chronos.agy_service import AgyBridgeService

                svc = AgyBridgeService(self.settings)
                if svc.available:
                    self._agy = svc
                    self.min_interval = 0.0
                    self.llm = self._make_agy_wrapper()
                else:
                    logger.warning(
                        "CHRONOS_PROCESSING_PROVIDER=agy but the AGY bridge token is unreadable; entity extraction disabled"
                    )
                    self.llm = None
            elif self._provider == "gemini":
                self.min_interval = (
                    max(60.0 / float(os.getenv("GEMINI_MAX_RPM", "10")), 0) + 0.5
                )
                if not self.settings.gemini_api_key:
                    logger.warning(
                        "CHRONOS_GEMINI_API_KEY not set; entity extraction disabled"
                    )
                    self.llm = None
                else:
                    self._gemini_client = get_genai_client()
                    self._gemini_model = self._select_gemini_model()
                    self.llm = self._make_gemini_wrapper()
            else:
                api_key = self.settings.openai_api_key
                if not api_key:
                    logger.warning(
                        "OpenAI entity extraction disabled by CHRONOS_OPENAI_ENABLED=0 or missing OPENAI_API_KEY"
                    )
                    self.llm = None
                else:
                    from openai import OpenAI

                    self._openai_client = OpenAI(api_key=api_key)
                    self._openai_model = os.getenv(
                        "CHRONOS_CLEANING_MODEL", "gpt-4.1-mini"
                    )
                    self.llm = self._make_openai_wrapper()
        else:
            self.llm = llm

        logger.info("✅ EntityExtractor initialized")

    def _select_gemini_model(self) -> str:
        configured = (
            getattr(self.settings, "chronos_cleaning_model", "") or ""
        ).strip()
        if configured.startswith("models/"):
            configured = configured.split("/", 1)[1]

        configured_candidate = configured if configured.startswith("gemini-") else ""
        selected = pick_first_available_or_known(
            configured_candidate,
            "gemini-2.5-flash",
            "gemini-3-flash-preview",
            "gemini-3.1-pro-preview",
        )
        return selected or "gemini-2.5-flash"

    def _make_openai_wrapper(self):
        """Create a simple wrapper matching the llama_index LLM interface."""
        client = self._openai_client
        model = self._openai_model

        class _OpenAIWrapper:
            def __init__(self):
                self.model = model

            def complete(self, prompt: str) -> _CompletionResult:
                kwargs = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_completion_tokens": 4096,
                }
                # Reasoning-family models (gpt-5.6 luna/terra/sol, o-series)
                # accept only the default temperature and reject an explicit
                # value with a 400. Sending it made extraction fail silently and
                # return no entities at all, so only set it where supported.
                if not _is_reasoning_model(model):
                    kwargs["temperature"] = 0.1
                response = client.chat.completions.create(**kwargs)
                usage = getattr(response, "usage", None)
                return _CompletionResult(
                    text=response.choices[0].message.content or "",
                    input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                    output_tokens=(
                        getattr(usage, "completion_tokens", 0) if usage else 0
                    ),
                )

        return _OpenAIWrapper()

    def _make_gemini_wrapper(self):
        """Create a Gemini wrapper matching the local .complete() interface."""
        client = self._gemini_client
        model = self._gemini_model

        class _GeminiWrapper:
            def __init__(self):
                self.model = model

            def complete(self, prompt: str) -> _CompletionResult:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "temperature": 0.1,
                    },
                )
                usage = getattr(response, "usage_metadata", None)
                return _CompletionResult(
                    text=(getattr(response, "text", "") or ""),
                    input_tokens=(
                        getattr(usage, "prompt_token_count", 0) if usage else 0
                    ),
                    output_tokens=(
                        (getattr(usage, "candidates_token_count", 0) or 0)
                        + (getattr(usage, "thoughts_token_count", 0) or 0)
                        if usage
                        else 0
                    ),
                )

        return _GeminiWrapper()

    def extract_entities(
        self,
        text: str,
        doc_id: str,
        max_text_chars: int = 8000,
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        Extract entities and relationships from text.

        Args:
            text: Transcript text to analyze
            doc_id: Document identifier for linking
            max_text_chars: Max chars to send to LLM

        Returns:
            Tuple of (entities, relationships)
        """
        try:
            from app_v2.services.xray import xray_log as _xlog
        except ImportError:
            _xlog = None

        if not self.llm:
            logger.warning("No LLM available for entity extraction")
            return [], []

        # Respect rate limits before issuing the call
        self._respect_rate_limit()

        # Truncate text if needed
        truncated_text = text[:max_text_chars]

        if _xlog:
            _xlog(
                "graph",
                "extract",
                f"Extracting entities from {doc_id} ({len(truncated_text)} chars)",
            )

        _t0 = time.monotonic()
        try:
            # Avoid str.format eating JSON braces; simply replace the placeholder
            prompt = self.EXTRACTION_PROMPT.replace("{text}", truncated_text)
            completion = self.llm.complete(prompt)
            response = completion.text.strip()
            self._last_call_ts = time.monotonic()
            from src.chronos.cost_tracker import track_usage

            track_usage(
                self.llm.model,
                "entity",
                input_tokens=getattr(completion, "input_tokens", 0)
                or int(len(prompt.split()) * 1.3),
                output_tokens=getattr(completion, "output_tokens", 0)
                or int(len(response.split()) * 1.3),
            )

            # Parse JSON from response
            # Handle markdown code blocks
            if response.startswith("```"):
                response = response.split("```", 2)[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()

            data = json.loads(response)

            entities, relationships = self._parse_entities_from_response(
                data, doc_id, len(truncated_text)
            )
            _elapsed = (time.monotonic() - _t0) * 1000
            if _xlog:
                _xlog(
                    "graph",
                    "extract",
                    f"Found {len(entities)} entities, {len(relationships)} relationships in {doc_id}",
                    duration_ms=round(_elapsed, 1),
                    detail=f"entities={len(entities)} rels={len(relationships)}",
                    level="perf",
                )
            return entities, relationships

            # Process people
            for person in data.get("people", []):
                name = person.get("name") if isinstance(person, dict) else person
                if name:
                    entity = Entity(
                        id=Entity.generate_id(name, EntityType.PERSON),
                        name=name,
                        entity_type=EntityType.PERSON,
                        metadata=(
                            {"role": person.get("role")}
                            if isinstance(person, dict)
                            else {}
                        ),
                    )
                    entities.append(entity)
                    relationships.append(
                        Relationship(
                            source_id=doc_id,
                            target_id=entity.id,
                            relation_type=RelationType.MENTIONS,
                        )
                    )

            # Process projects
            for project in data.get("projects", []):
                name = project.get("name") if isinstance(project, dict) else project
                if name:
                    entity = Entity(
                        id=Entity.generate_id(name, EntityType.PROJECT),
                        name=name,
                        entity_type=EntityType.PROJECT,
                        metadata=(
                            {"status": project.get("status")}
                            if isinstance(project, dict)
                            else {}
                        ),
                    )
                    entities.append(entity)
                    relationships.append(
                        Relationship(
                            source_id=doc_id,
                            target_id=entity.id,
                            relation_type=RelationType.MENTIONS,
                        )
                    )

            # Process topics
            for topic in data.get("topics", []):
                if topic:
                    entity = Entity(
                        id=Entity.generate_id(topic, EntityType.TOPIC),
                        name=topic,
                        entity_type=EntityType.TOPIC,
                    )
                    entities.append(entity)
                    relationships.append(
                        Relationship(
                            source_id=doc_id,
                            target_id=entity.id,
                            relation_type=RelationType.DISCUSSED_IN,
                        )
                    )

            # Process actions
            for action in data.get("actions", []):
                task = action.get("task") if isinstance(action, dict) else action
                if task:
                    entity = Entity(
                        id=Entity.generate_id(task[:50], EntityType.ACTION),
                        name=task,
                        entity_type=EntityType.ACTION,
                        metadata=(
                            {
                                "assignee": action.get("assignee"),
                                "deadline": action.get("deadline"),
                            }
                            if isinstance(action, dict)
                            else {}
                        ),
                    )
                    entities.append(entity)

                    # Link action to document
                    relationships.append(
                        Relationship(
                            source_id=doc_id,
                            target_id=entity.id,
                            relation_type=RelationType.MENTIONS,
                        )
                    )

                    # Link action to assignee if present
                    if isinstance(action, dict) and action.get("assignee"):
                        assignee_id = Entity.generate_id(
                            action["assignee"], EntityType.PERSON
                        )
                        relationships.append(
                            Relationship(
                                source_id=entity.id,
                                target_id=assignee_id,
                                relation_type=RelationType.ASSIGNED_TO,
                            )
                        )

            # Process organizations
            for org in data.get("organizations", []):
                if org:
                    entity = Entity(
                        id=Entity.generate_id(org, EntityType.ORGANIZATION),
                        name=org,
                        entity_type=EntityType.ORGANIZATION,
                    )
                    entities.append(entity)
                    relationships.append(
                        Relationship(
                            source_id=doc_id,
                            target_id=entity.id,
                            relation_type=RelationType.MENTIONS,
                        )
                    )

            logger.info(
                f"   📊 Extracted {len(entities)} entities, {len(relationships)} relationships"
            )
            return entities, relationships

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse entity extraction response: {e}")
            return [], []
        except Exception as e:
            err_text = str(e)

            if "429" in err_text or "quota" in err_text.lower():
                retry_delay = self._parse_retry_delay(
                    err_text, default=self.min_interval
                )
                logger.warning(
                    f"Entity extraction hit rate limit; sleeping {retry_delay:.1f}s then retrying once"
                )
                time.sleep(retry_delay)
                try:
                    self._respect_rate_limit()
                    completion = self.llm.complete(prompt)
                    response = completion.text.strip()
                    self._last_call_ts = time.monotonic()
                    from src.chronos.cost_tracker import track_usage

                    track_usage(
                        self.llm.model,
                        "entity",
                        input_tokens=getattr(completion, "input_tokens", 0)
                        or int(len(prompt.split()) * 1.3),
                        output_tokens=getattr(completion, "output_tokens", 0)
                        or int(len(response.split()) * 1.3),
                    )
                    if response.startswith("```"):
                        response = response.split("```", 2)[1]
                        if response.startswith("json"):
                            response = response[4:]
                        response = response.strip()
                    data = json.loads(response)
                    return self._parse_entities_from_response(
                        data, doc_id, len(truncated_text)
                    )
                except Exception as e2:
                    logger.error(
                        f"Entity extraction retry failed: {e2}. Snippet: {str(e2)[:400]}"
                    )
                    return [], []

            logger.error(f"Entity extraction failed: {e}")
            return [], []

    def _make_agy_wrapper(self):
        """AGY bridge wrapper matching the .complete() interface (one event per call)."""
        from src.chronos.agy_service import parse_json_objects

        svc = self._agy

        class _AgyWrapper:
            def __init__(self):
                self.model = f"agy/{svc.model}"

            def complete(self, prompt: str) -> _CompletionResult:
                result = svc.complete(prompt, "Return only the JSON object.")
                if not result.get("ok"):
                    raise RuntimeError(f"AGY bridge: {result.get('error')}")
                text = result.get("text") or ""
                found = parse_json_objects(text)
                usage = result.get("usage") or {}
                return _CompletionResult(
                    # agy can return the object twice; hand back the last one
                    text=json.dumps(found[-1]) if found else text,
                    input_tokens=int(usage.get("input_tokens") or 0),
                    output_tokens=int(usage.get("output_tokens") or 0),
                )

        return _AgyWrapper()

    @property
    def supports_batch(self) -> bool:
        """True when many events can share one model call (the AGY path)."""
        return self._agy is not None and self.llm is not None

    BATCH_PROMPT = """Extract entities from EACH event below. Every event is a cleaned moment from the
owner's voice recordings, introduced by a line "=== EVENT <event_id> ===".

For every event return one object with its exact event_id and these lists:
people [{name, role}], projects [{name, status}], topics [strings], actions [{task, assignee, deadline}],
dates [strings], metrics [{value, context}], organizations [strings], locations [strings],
relationships [{source, source_type, relation, target, target_type, evidence}].

RELATIONSHIP RULES:
- Only relationships stated or clearly implied in THAT event, between named entities you listed above.
- source_type/target_type: person, project, topic, organization, location.
- relation: works_with, works_on, member_of, reports_to, assigned_to, part_of, uses, located_in, knows, related_to.
  (knows = personal tie such as friend or family; related_to only when nothing more specific fits.)
- evidence: at most 15 words copied from the event that show the relationship.

CRITICAL TOPIC RULES:
- Topics MUST be concrete subject nouns, proper nouns, technologies, projects, or multi-word concept phrases (e.g., "Raspberry Pi", "Notion Sync", "API Billing", "iOS App", "Tailscale").
- NEVER extract single verbs, action words, gerunds, or conversational filler as topics (e.g. NEVER output "going", "using", "swapping", "talking", "asking", "doing", "running", "wants").
- Every topic must be a distinct, meaningful subject entity.

Only include entities clearly mentioned in THAT event. Be specific with names. Use empty lists when
nothing is found. Include every event_id exactly once, even when all its lists are empty.
"""

    @staticmethod
    def _batch_schema() -> Dict[str, Any]:
        def obj(*keys):
            return {"type": "object", "properties": {k: {"type": "string"} for k in keys}}

        strings = {"type": "array", "items": {"type": "string"}}
        item = {
            "type": "object",
            "properties": {
                "event_id": {"type": "string"},
                "people": {"type": "array", "items": obj("name", "role")},
                "projects": {"type": "array", "items": obj("name", "status")},
                "topics": strings,
                "actions": {"type": "array", "items": obj("task", "assignee", "deadline")},
                "dates": strings,
                "metrics": {"type": "array", "items": obj("value", "context")},
                "organizations": strings,
                "locations": strings,
                "relationships": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "source_type": {"type": "string", "enum": sorted(LINKABLE_TYPES)},
                            "relation": {"type": "string", "enum": sorted(STATED_RELATIONS)},
                            "target": {"type": "string"},
                            "target_type": {"type": "string", "enum": sorted(LINKABLE_TYPES)},
                            "evidence": {"type": "string"},
                        },
                        "required": ["source", "source_type", "relation", "target", "target_type"],
                    },
                },
            },
            "required": ["event_id"],
        }
        return {
            "type": "object",
            "properties": {"events": {"type": "array", "items": item}},
            "required": ["events"],
        }

    def extract_entities_batch(
        self,
        items: List[Tuple[str, str]],
        max_text_chars: int = 4000,
    ) -> Dict[str, Tuple[List[Entity], List[Relationship]]]:
        """Extract entities for many (doc_id, text) pairs in ONE AGY call.

        Returns {doc_id: (entities, relationships)} for every event the model answered;
        ids it skipped are simply absent. Raises on a bridge failure.
        """
        from src.chronos.agy_service import parse_json_objects
        from src.chronos.cost_tracker import track_usage

        if not self.supports_batch:
            raise RuntimeError("batch extraction needs the AGY provider")
        texts = {doc_id: (text or "")[:max_text_chars] for doc_id, text in items}
        blocks = "".join(f"\n=== EVENT {doc_id} ===\n{text}\n" for doc_id, text in texts.items())
        _t0 = time.monotonic()
        result = self._agy.complete(
            self.BATCH_PROMPT + blocks,
            f"Return only the JSON object with one entry per event ({len(texts)} events).",
            self._batch_schema(),
        )
        if not result.get("ok"):
            raise RuntimeError(f"AGY bridge: {result.get('error')}")
        usage = result.get("usage") or {}
        track_usage(
            self.llm.model,
            "entity",
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
        )

        answers: List[Dict[str, Any]] = []
        for candidate in reversed(parse_json_objects(result.get("text") or "")):
            if isinstance(candidate.get("events"), list):
                answers = [a for a in candidate["events"] if isinstance(a, dict)]
                break
        out: Dict[str, Tuple[List[Entity], List[Relationship]]] = {}
        for answer in answers:
            doc_id = str(answer.get("event_id") or "")
            if doc_id in texts and doc_id not in out:
                out[doc_id] = self._parse_entities_from_response(answer, doc_id, len(texts[doc_id]))
        logger.info(
            "AGY entity batch: %d/%d events answered in %.0fs",
            len(out), len(texts), time.monotonic() - _t0,
        )
        return out

    def _respect_rate_limit(self) -> None:
        """Sleep if the previous call was too recent."""
        if self._last_call_ts <= 0:
            return
        elapsed = time.monotonic() - self._last_call_ts
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def _parse_retry_delay(self, err_text: str, default: float) -> float:
        """Extract retry delay seconds from Gemini error if present."""
        match = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", err_text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return default
        return default

    def _parse_entities_from_response(
        self, data: Dict[str, Any], doc_id: str, text_len: int
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Parse entities/relationships from a JSON response payload."""
        entities: List[Entity] = []
        relationships: List[Relationship] = []

        # Process people
        for person in data.get("people", []):
            name = person.get("name") if isinstance(person, dict) else person
            if name:
                entity = Entity(
                    id=Entity.generate_id(name, EntityType.PERSON),
                    name=name,
                    entity_type=EntityType.PERSON,
                    metadata=(
                        {"role": person.get("role")} if isinstance(person, dict) else {}
                    ),
                )
                entities.append(entity)
                relationships.append(
                    Relationship(
                        source_id=doc_id,
                        target_id=entity.id,
                        relation_type=RelationType.MENTIONS,
                    )
                )

        # Process projects
        for project in data.get("projects", []):
            name = project.get("name") if isinstance(project, dict) else project
            if name:
                entity = Entity(
                    id=Entity.generate_id(name, EntityType.PROJECT),
                    name=name,
                    entity_type=EntityType.PROJECT,
                    metadata=(
                        {"status": project.get("status")}
                        if isinstance(project, dict)
                        else {}
                    ),
                )
                entities.append(entity)
                relationships.append(
                    Relationship(
                        source_id=doc_id,
                        target_id=entity.id,
                        relation_type=RelationType.MENTIONS,
                    )
                )

        # Process topics
        for topic in data.get("topics", []):
            if topic:
                entity = Entity(
                    id=Entity.generate_id(topic, EntityType.TOPIC),
                    name=topic,
                    entity_type=EntityType.TOPIC,
                )
                entities.append(entity)
                relationships.append(
                    Relationship(
                        source_id=doc_id,
                        target_id=entity.id,
                        relation_type=RelationType.DISCUSSED_IN,
                    )
                )

        # Process actions
        for action in data.get("actions", []):
            task = action.get("task") if isinstance(action, dict) else action
            if task:
                entity = Entity(
                    id=Entity.generate_id(task[:50], EntityType.ACTION),
                    name=task,
                    entity_type=EntityType.ACTION,
                    metadata=(
                        {
                            "assignee": action.get("assignee"),
                            "deadline": action.get("deadline"),
                        }
                        if isinstance(action, dict)
                        else {}
                    ),
                )
                entities.append(entity)

                # Link action to document
                relationships.append(
                    Relationship(
                        source_id=doc_id,
                        target_id=entity.id,
                        relation_type=RelationType.MENTIONS,
                    )
                )

                # Link action to assignee if present
                if isinstance(action, dict) and action.get("assignee"):
                    assignee_id = Entity.generate_id(
                        action["assignee"], EntityType.PERSON
                    )
                    relationships.append(
                        Relationship(
                            source_id=entity.id,
                            target_id=assignee_id,
                            relation_type=RelationType.ASSIGNED_TO,
                        )
                    )

        # Process organizations
        for org in data.get("organizations", []):
            if org:
                entity = Entity(
                    id=Entity.generate_id(org, EntityType.ORGANIZATION),
                    name=org,
                    entity_type=EntityType.ORGANIZATION,
                )
                entities.append(entity)
                relationships.append(
                    Relationship(
                        source_id=doc_id,
                        target_id=entity.id,
                        relation_type=RelationType.MENTIONS,
                    )
                )

        # Process locations
        for place in data.get("locations", []) or []:
            if place and isinstance(place, str):
                entity = Entity(
                    id=Entity.generate_id(place, EntityType.LOCATION),
                    name=place,
                    entity_type=EntityType.LOCATION,
                )
                entities.append(entity)
                relationships.append(
                    Relationship(
                        source_id=doc_id,
                        target_id=entity.id,
                        relation_type=RelationType.MENTIONS,
                    )
                )

        # Typed entity-to-entity relationships the model stated (with evidence)
        known = {e.id for e in entities}
        for rel in data.get("relationships", []) or []:
            if not isinstance(rel, dict):
                continue
            relation = str(rel.get("relation") or "").strip().lower()
            source_type = str(rel.get("source_type") or "").strip().lower()
            target_type = str(rel.get("target_type") or "").strip().lower()
            source = str(rel.get("source") or "").strip()
            target = str(rel.get("target") or "").strip()
            if (
                relation not in STATED_RELATIONS
                or source_type not in LINKABLE_TYPES
                or target_type not in LINKABLE_TYPES
                or not source
                or not target
            ):
                continue
            ends = []
            for name, kind in ((source, EntityType(source_type)), (target, EntityType(target_type))):
                entity_id = Entity.generate_id(name, kind)
                if entity_id not in known:  # the model named it only inside the relationship
                    entities.append(Entity(id=entity_id, name=name, entity_type=kind))
                    known.add(entity_id)
                ends.append(entity_id)
            if ends[0] == ends[1]:
                continue
            evidence = str(rel.get("evidence") or "").strip()
            relationships.append(
                Relationship(
                    source_id=ends[0],
                    target_id=ends[1],
                    relation_type=RelationType(relation),
                    metadata={"evidence": [evidence[:200]]} if evidence else {},
                )
            )

        logger.info(
            f"   📊 Extracted {len(entities)} entities, {len(relationships)} relationships"
        )
        if len(entities) == 0:
            if text_len >= _NO_ENTITY_WARNING_MIN_TEXT_LEN:
                logger.warning(
                    f"[GraphRAG] No entities extracted for doc {doc_id}. Text len={text_len}"
                )
            else:
                logger.debug(
                    f"[GraphRAG] Skipping no-entity warning for short doc {doc_id}. Text len={text_len}"
                )
        return entities, relationships


# Global knowledge graph instance
_knowledge_graph: Optional[KnowledgeGraph] = None


def get_knowledge_graph() -> KnowledgeGraph:
    """Get or create the global knowledge graph."""
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = KnowledgeGraph()
    return _knowledge_graph


# ============================================================================
# COMMUNITY SUMMARIZATION (GraphRAG Enhancement)
# ============================================================================
# Reference: gemini-deep-research2.txt - Microsoft GraphRAG with Leiden algorithm
#
# Community detection clusters related entities to answer GLOBAL queries like:
# - "What are the main themes across all recordings?"
# - "Summarize all discussions about the product roadmap"
# - "What topics were most discussed this quarter?"
#
# These queries FAIL with pure vector search because no single document
# contains the answer - it emerges from aggregating the entire corpus.


@dataclass
class Community:
    """A cluster of related entities in the knowledge graph."""

    id: str
    entities: List[str]  # Entity IDs in this community
    summary: str = ""  # LLM-generated summary
    keywords: List[str] = field(default_factory=list)
    document_count: int = 0  # Documents touching this community

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "entity_count": len(self.entities),
            "summary": self.summary,
            "keywords": self.keywords,
            "document_count": self.document_count,
        }


class CommunityDetector:
    """
    Detects communities (clusters) in the knowledge graph using a
    simplified Louvain-style algorithm (NetworkX if available, else greedy).

    Communities enable answering GLOBAL queries that require synthesis
    across the entire corpus rather than retrieval of single documents.

    Reference: Microsoft GraphRAG uses Leiden algorithm, we use Louvain
    which is simpler and has better Python library support.
    """

    SUMMARY_PROMPT = """Summarize this cluster of related entities and their relationships in 2-3 sentences.

Entities in cluster:
{entities}

Key relationships:
{relationships}

Write a concise summary describing what this cluster represents (e.g., "A project team working on X" or "Discussions about budget and timeline").
Also provide 3-5 keywords that capture the essence of this cluster.

Respond in JSON:
{{"summary": "...", "keywords": ["keyword1", "keyword2", "keyword3"]}}"""

    def __init__(self, min_community_size: int = 2):
        """
        Initialize community detector.

        Args:
            min_community_size: Minimum entities per community (default 2)
        """
        self.min_size = min_community_size
        self._llm = None

    def _get_llm(self):
        """Lazy load OpenAI client + model for summarization."""
        if self._llm is None:
            from src.config import get_settings

            settings = get_settings()
            if settings.openai_api_key:
                from openai import OpenAI

                client = OpenAI(api_key=settings.openai_api_key)
                model_name = settings.chronos_cleaning_model
                self._llm = (client, model_name)
        return self._llm

    def detect_communities(self, graph: KnowledgeGraph) -> List[Community]:
        """
        Detect communities in the knowledge graph.

        Uses NetworkX's Louvain algorithm if available, else falls back
        to simple connected components.

        Args:
            graph: KnowledgeGraph instance

        Returns:
            List of Community objects
        """
        try:
            import networkx as nx
            from networkx.algorithms.community import louvain_communities

            return self._detect_with_networkx(graph)
        except ImportError:
            logger.warning("NetworkX not available, using simple clustering")
            return self._detect_simple(graph)

    def _detect_with_networkx(self, graph: KnowledgeGraph) -> List[Community]:
        """Use NetworkX Louvain community detection."""
        import networkx as nx
        from networkx.algorithms.community import louvain_communities

        # Build NetworkX graph
        G = nx.Graph()

        # Add nodes (entities)
        for entity_id, entity in graph.entities.items():
            G.add_node(entity_id, name=entity.name, type=entity.entity_type.value)

        # Add edges (relationships)
        for rel in graph.relationships:
            if rel.source_id in G.nodes and rel.target_id in G.nodes:
                G.add_edge(
                    rel.source_id,
                    rel.target_id,
                    weight=rel.weight,
                    type=rel.relation_type.value,
                )

        # Detect communities
        communities = louvain_communities(G, seed=42)

        # Convert to Community objects
        result = []
        for i, community_set in enumerate(communities):
            entity_ids = list(community_set)
            if len(entity_ids) >= self.min_size:
                community = Community(
                    id=f"community_{i}",
                    entities=entity_ids,
                )
                # Count documents
                doc_ids = set()
                for eid in entity_ids:
                    doc_ids.update(graph.get_related_documents(eid))
                community.document_count = len(doc_ids)

                result.append(community)

        logger.info(
            f"🔗 Detected {len(result)} communities from {len(graph.entities)} entities"
        )
        try:
            from app_v2.services.xray import xray_log

            xray_log(
                "graph",
                "communities",
                f"Louvain detected {len(result)} communities from {len(graph.entities)} entities ({G.number_of_edges()} edges)",
                detail=f"algorithm=louvain entities={len(graph.entities)} communities={len(result)}",
            )
        except ImportError:
            pass
        return result

    def _detect_simple(self, graph: KnowledgeGraph) -> List[Community]:
        """Simple connected components clustering (fallback)."""
        # Build adjacency
        adjacency: Dict[str, Set[str]] = {eid: set() for eid in graph.entities}
        for rel in graph.relationships:
            if rel.source_id in adjacency and rel.target_id in adjacency:
                adjacency[rel.source_id].add(rel.target_id)
                adjacency[rel.target_id].add(rel.source_id)

        # Find connected components via BFS
        visited = set()
        components = []

        for start_id in graph.entities:
            if start_id in visited:
                continue

            component = []
            queue = [start_id]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                queue.extend(n for n in adjacency[node] if n not in visited)

            if len(component) >= self.min_size:
                components.append(component)

        # Convert to Community objects
        result = []
        for i, entity_ids in enumerate(components):
            community = Community(id=f"community_{i}", entities=entity_ids)
            doc_ids = set()
            for eid in entity_ids:
                doc_ids.update(graph.get_related_documents(eid))
            community.document_count = len(doc_ids)
            result.append(community)

        return result

    def summarize_community(
        self, community: Community, graph: KnowledgeGraph
    ) -> Community:
        """
        Generate LLM summary for a community.

        Args:
            community: Community to summarize
            graph: KnowledgeGraph for entity/relationship details

        Returns:
            Community with summary and keywords filled in
        """
        llm = self._get_llm()
        if not llm:
            community.summary = f"Cluster of {len(community.entities)} related entities"
            return community

        client, model_name = llm

        # Build entity list
        entities_text = []
        for eid in community.entities[:20]:  # Limit for prompt size
            entity = graph.entities.get(eid)
            if entity:
                entities_text.append(f"- {entity.name} ({entity.entity_type.value})")

        # Build relationship list
        rels_text = []
        community_set = set(community.entities)
        for rel in graph.relationships:
            if rel.source_id in community_set and rel.target_id in community_set:
                source = graph.entities.get(rel.source_id)
                target = graph.entities.get(rel.target_id)
                if source and target:
                    rels_text.append(
                        f"- {source.name} --[{rel.relation_type.value}]--> {target.name}"
                    )

        prompt = self.SUMMARY_PROMPT.format(
            entities="\n".join(entities_text[:20]),
            relationships="\n".join(rels_text[:15]),
        )

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_completion_tokens=2048,
            )
            # Track cost
            _usage = response.usage
            if _usage:
                from src.chronos.cost_tracker import track_usage

                track_usage(
                    model_name,
                    "community",
                    input_tokens=getattr(_usage, "prompt_tokens", 0),
                    output_tokens=getattr(_usage, "completion_tokens", 0),
                )
            text = (response.choices[0].message.content or "").strip()

            # Parse JSON
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            data = json.loads(text)
            community.summary = data.get("summary", "")
            community.keywords = data.get("keywords", [])
            try:
                from app_v2.services.xray import xray_log

                xray_log(
                    "graph",
                    "summarize",
                    f"Summarized community {community.id}: {len(community.entities)} entities → {len(community.keywords)} keywords",
                    detail=f"model={model_name} keywords={','.join(community.keywords[:5])}",
                )
            except ImportError:
                pass

        except Exception as e:
            logger.warning(f"Community summarization failed: {e}")
            community.summary = f"Cluster containing: {', '.join(e.name for e in (graph.entities.get(eid) for eid in community.entities[:5]) if e)}"

        return community


# Global community cache
_community_cache: Optional[List[Community]] = None


def detect_and_summarize_communities(force_refresh: bool = False) -> List[Community]:
    """
    Detect communities in the global knowledge graph and generate summaries.

    This enables answering GLOBAL queries like:
    - "What are the main themes across all recordings?"
    - "Summarize all discussions about budget"

    Args:
        force_refresh: Force re-detection even if cached

    Returns:
        List of summarized Community objects
    """
    global _community_cache

    if _community_cache is not None and not force_refresh:
        return _community_cache

    graph = get_knowledge_graph()
    if not graph.entities:
        return []

    detector = CommunityDetector()
    communities = detector.detect_communities(graph)

    # Summarize each community
    for community in communities:
        detector.summarize_community(community, graph)

    _community_cache = communities

    logger.info(f"📊 Generated {len(communities)} community summaries")
    return communities


def answer_global_query(query: str) -> Dict:
    """
    Answer a GLOBAL query using community summaries.

    This is for queries that require synthesis across the entire corpus,
    not retrieval of specific documents.

    Examples:
    - "What are the main themes discussed?"
    - "Summarize all budget-related discussions"
    - "What topics were most common this quarter?"

    Args:
        query: Global/aggregation query

    Returns:
        Dict with relevant communities and synthesized answer
    """
    communities = detect_and_summarize_communities()

    if not communities:
        return {
            "query": query,
            "answer": "No community summaries available. Process some documents first.",
            "communities": [],
        }

    # Find relevant communities by keyword matching
    query_lower = query.lower()
    scored_communities = []

    for community in communities:
        score = 0
        # Match keywords
        for keyword in community.keywords:
            if keyword.lower() in query_lower:
                score += 2
        # Match summary content
        if community.summary:
            words = query_lower.split()
            for word in words:
                if len(word) > 3 and word in community.summary.lower():
                    score += 1

        if score > 0:
            scored_communities.append((community, score))

    # Sort by relevance
    scored_communities.sort(key=lambda x: x[1], reverse=True)
    top_communities = [c for c, s in scored_communities[:5]]

    # Synthesize answer from community summaries
    if top_communities:
        summaries = [c.summary for c in top_communities if c.summary]
        combined = " ".join(summaries)
        answer = f"Based on {len(top_communities)} related topic clusters: {combined}"
    else:
        # Return overview of all communities
        all_summaries = [c.summary for c in communities[:3] if c.summary]
        answer = f"Main themes across recordings: {' '.join(all_summaries)}"

    return {
        "query": query,
        "answer": answer,
        "communities": [c.to_dict() for c in (top_communities or communities[:3])],
        "total_communities": len(communities),
    }


# Backward compatibility alias
GraphRAGExtractor = EntityExtractor
