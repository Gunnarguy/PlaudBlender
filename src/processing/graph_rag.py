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
import json
import logging
import hashlib
from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


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
    MENTIONS = "mentions"           # Document mentions entity
    RELATED_TO = "related_to"       # Entity is related to another
    ASSIGNED_TO = "assigned_to"     # Action assigned to person
    DISCUSSED_IN = "discussed_in"   # Topic discussed in meeting
    WORKS_ON = "works_on"           # Person works on project
    REPORTS_TO = "reports_to"       # Person reports to another
    DEADLINE = "deadline"           # Action has deadline


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
    def generate_id(name: str, entity_type: EntityType) -> str:
        """Generate deterministic entity ID."""
        content = f"{entity_type.value}:{name.lower().strip()}"
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
    document_entities: Dict[str, Set[str]] = field(default_factory=dict)  # doc_id -> entity_ids
    
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
        for existing in self.relationships:
            if (existing.source_id == rel.source_id and 
                existing.target_id == rel.target_id and 
                existing.relation_type == rel.relation_type):
                existing.weight += rel.weight
                return
        self.relationships.append(rel)
    
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
        for rel in self.relationships:
            if rel.source_id == entity_id:
                target = self.entities.get(rel.target_id)
                if target:
                    neighbors.append((target, rel))
            elif rel.target_id == entity_id:
                source = self.entities.get(rel.source_id)
                if source:
                    neighbors.append((source, rel))
        return neighbors
    
    def search_entities(self, query: str, entity_type: Optional[EntityType] = None) -> List[Entity]:
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
    
    EXTRACTION_PROMPT = '''Extract entities from this transcript text. Return a JSON object with the following structure:

{
  "people": [{"name": "Person Name", "role": "optional role/title"}],
  "projects": [{"name": "Project Name", "status": "optional status"}],
  "topics": ["Topic 1", "Topic 2"],
  "actions": [{"task": "Task description", "assignee": "Person (if mentioned)", "deadline": "Date (if mentioned)"}],
  "dates": ["2024-10-15", "Q3 2024"],
  "metrics": [{"value": "15%", "context": "revenue growth"}],
  "organizations": ["Company/Org Name"]
}

Only include entities that are clearly mentioned. Be specific with names.
If an entity is not found, use an empty array.

Transcript:
{text}

Return ONLY the JSON object, no other text.'''

    def __init__(self, llm=None):
        """
        Initialize entity extractor.
        
        Args:
            llm: LLM instance (defaults to Gemini)
        """
        if llm is None:
            try:
                from llama_index.llms.gemini import Gemini
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("GEMINI_API_KEY required for entity extraction")
                self.llm = Gemini(
                    model="models/gemini-2.0-flash-exp",
                    api_key=api_key,
                    temperature=0.1,  # Low temp for structured extraction
                )
            except ImportError:
                logger.warning("llama_index.llms.gemini not available")
                self.llm = None
        else:
            self.llm = llm
        
        logger.info("✅ EntityExtractor initialized")
    
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
        if not self.llm:
            logger.warning("No LLM available for entity extraction")
            return [], []
        
        # Truncate text if needed
        truncated_text = text[:max_text_chars]
        
        try:
            prompt = self.EXTRACTION_PROMPT.format(text=truncated_text)
            response = self.llm.complete(prompt).text.strip()
            
            # Parse JSON from response
            # Handle markdown code blocks
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()
            
            data = json.loads(response)
            
            entities = []
            relationships = []
            
            # Process people
            for person in data.get("people", []):
                name = person.get("name") if isinstance(person, dict) else person
                if name:
                    entity = Entity(
                        id=Entity.generate_id(name, EntityType.PERSON),
                        name=name,
                        entity_type=EntityType.PERSON,
                        metadata={"role": person.get("role")} if isinstance(person, dict) else {},
                    )
                    entities.append(entity)
                    relationships.append(Relationship(
                        source_id=doc_id,
                        target_id=entity.id,
                        relation_type=RelationType.MENTIONS,
                    ))
            
            # Process projects
            for project in data.get("projects", []):
                name = project.get("name") if isinstance(project, dict) else project
                if name:
                    entity = Entity(
                        id=Entity.generate_id(name, EntityType.PROJECT),
                        name=name,
                        entity_type=EntityType.PROJECT,
                        metadata={"status": project.get("status")} if isinstance(project, dict) else {},
                    )
                    entities.append(entity)
                    relationships.append(Relationship(
                        source_id=doc_id,
                        target_id=entity.id,
                        relation_type=RelationType.MENTIONS,
                    ))
            
            # Process topics
            for topic in data.get("topics", []):
                if topic:
                    entity = Entity(
                        id=Entity.generate_id(topic, EntityType.TOPIC),
                        name=topic,
                        entity_type=EntityType.TOPIC,
                    )
                    entities.append(entity)
                    relationships.append(Relationship(
                        source_id=doc_id,
                        target_id=entity.id,
                        relation_type=RelationType.DISCUSSED_IN,
                    ))
            
            # Process actions
            for action in data.get("actions", []):
                task = action.get("task") if isinstance(action, dict) else action
                if task:
                    entity = Entity(
                        id=Entity.generate_id(task[:50], EntityType.ACTION),
                        name=task,
                        entity_type=EntityType.ACTION,
                        metadata={
                            "assignee": action.get("assignee"),
                            "deadline": action.get("deadline"),
                        } if isinstance(action, dict) else {},
                    )
                    entities.append(entity)
                    
                    # Link action to document
                    relationships.append(Relationship(
                        source_id=doc_id,
                        target_id=entity.id,
                        relation_type=RelationType.MENTIONS,
                    ))
                    
                    # Link action to assignee if present
                    if isinstance(action, dict) and action.get("assignee"):
                        assignee_id = Entity.generate_id(action["assignee"], EntityType.PERSON)
                        relationships.append(Relationship(
                            source_id=entity.id,
                            target_id=assignee_id,
                            relation_type=RelationType.ASSIGNED_TO,
                        ))
            
            # Process organizations
            for org in data.get("organizations", []):
                if org:
                    entity = Entity(
                        id=Entity.generate_id(org, EntityType.ORGANIZATION),
                        name=org,
                        entity_type=EntityType.ORGANIZATION,
                    )
                    entities.append(entity)
                    relationships.append(Relationship(
                        source_id=doc_id,
                        target_id=entity.id,
                        relation_type=RelationType.MENTIONS,
                    ))
            
            logger.info(f"   📊 Extracted {len(entities)} entities, {len(relationships)} relationships")
            return entities, relationships
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse entity extraction response: {e}")
            return [], []
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return [], []


# Global knowledge graph instance
_knowledge_graph: Optional[KnowledgeGraph] = None


def get_knowledge_graph() -> KnowledgeGraph:
    """Get or create the global knowledge graph."""
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = KnowledgeGraph()
    return _knowledge_graph


def extract_and_store(
    text: str,
    doc_id: str,
    extractor: Optional[EntityExtractor] = None,
) -> Tuple[int, int]:
    """
    Extract entities from text and store in global knowledge graph.
    
    Args:
        text: Transcript text
        doc_id: Document identifier
        extractor: EntityExtractor instance (creates one if None)
    
    Returns:
        Tuple of (entities_added, relationships_added)
    """
    if extractor is None:
        extractor = EntityExtractor()
    
    entities, relationships = extractor.extract_entities(text, doc_id)
    
    graph = get_knowledge_graph()
    entity_ids = []
    
    for entity in entities:
        graph.add_entity(entity)
        entity_ids.append(entity.id)
    
    for rel in relationships:
        graph.add_relationship(rel)
    
    graph.link_document(doc_id, entity_ids)
    
    return len(entities), len(relationships)


def query_graph(
    query: str,
    entity_type: Optional[EntityType] = None,
    hop_depth: int = 1,
) -> Dict:
    """
    Query the knowledge graph for entities and their connections.
    
    Args:
        query: Search query (matches entity names)
        entity_type: Filter by entity type
        hop_depth: How many relationship hops to traverse
    
    Returns:
        Dict with matching entities and related entities
    """
    graph = get_knowledge_graph()
    
    # Find matching entities
    matches = graph.search_entities(query, entity_type)
    
    result = {
        "query": query,
        "matches": [e.to_dict() for e in matches[:10]],
        "related": [],
        "documents": [],
    }
    
    # Get related entities (1-hop)
    if matches and hop_depth >= 1:
        related_set = set()
        for match in matches[:5]:
            neighbors = graph.get_entity_neighbors(match.id)
            for entity, rel in neighbors:
                if entity.id not in related_set:
                    related_set.add(entity.id)
                    result["related"].append({
                        "entity": entity.to_dict(),
                        "relation": rel.relation_type.value,
                    })
    
    # Get related documents
    for match in matches[:5]:
        docs = graph.get_related_documents(match.id)
        for doc in docs:
            if doc not in result["documents"]:
                result["documents"].append(doc)
    
    return result
