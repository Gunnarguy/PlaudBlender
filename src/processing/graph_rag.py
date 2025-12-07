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
    entities: List[str]           # Entity IDs in this community
    summary: str = ""             # LLM-generated summary
    keywords: List[str] = field(default_factory=list)
    document_count: int = 0       # Documents touching this community
    
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
    
    SUMMARY_PROMPT = '''Summarize this cluster of related entities and their relationships in 2-3 sentences.

Entities in cluster:
{entities}

Key relationships:
{relationships}

Write a concise summary describing what this cluster represents (e.g., "A project team working on X" or "Discussions about budget and timeline").
Also provide 3-5 keywords that capture the essence of this cluster.

Respond in JSON:
{{"summary": "...", "keywords": ["keyword1", "keyword2", "keyword3"]}}'''

    def __init__(self, min_community_size: int = 2):
        """
        Initialize community detector.
        
        Args:
            min_community_size: Minimum entities per community (default 2)
        """
        self.min_size = min_community_size
        self._llm = None
    
    def _get_llm(self):
        """Lazy load LLM for summarization."""
        if self._llm is None:
            import google.generativeai as genai
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                self._llm = genai.GenerativeModel("gemini-1.5-flash")
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
            G.add_node(entity_id, 
                       name=entity.name, 
                       type=entity.entity_type.value)
        
        # Add edges (relationships)
        for rel in graph.relationships:
            if rel.source_id in G.nodes and rel.target_id in G.nodes:
                G.add_edge(rel.source_id, rel.target_id, 
                           weight=rel.weight,
                           type=rel.relation_type.value)
        
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
        
        logger.info(f"🔗 Detected {len(result)} communities from {len(graph.entities)} entities")
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
    
    def summarize_community(self, community: Community, graph: KnowledgeGraph) -> Community:
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
            response = llm.generate_content(prompt)
            text = response.text.strip()
            
            # Parse JSON
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            data = json.loads(text)
            community.summary = data.get("summary", "")
            community.keywords = data.get("keywords", [])
            
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
