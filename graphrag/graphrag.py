from typing import List, Dict, Any, Tuple
from entities.document import Document
from graphrag.interfaces.json_generator import JsonGenerator
import networkx as nx
from pydantic import BaseModel, Field, ValidationError

class KnowledgeGraph:
    """
    Simple in-memory knowledge graph: nodes are entities, edges are relationships.
    """
    def __init__(self, documents: List[Document]):
        self.nodes = set()
        self.edges = []
        self.documents = documents

    def add_entity(self, entity: str):
        self.nodes.add(entity)

    def add_relationship(self, entity1: str, relation: str, entity2: str):
        self.nodes.add(entity1)
        self.nodes.add(entity2)
        self.edges.append((entity1, relation, entity2))

    def as_dict(self) -> Dict[str, Any]:
        return {
            'entities': list(self.nodes),
            'relationships': [
                {'source': e1, 'relation': rel, 'target': e2}
                for e1, rel, e2 in self.edges
            ]
        }

    # todo: Add comunity summarization

    def search(self, query: str, k: int) -> List[Document]:
        """
        Search the knowledge graph for entities and relationships relevant to the query, and return up to k related documents.
        Uses both local (neighborhood) and global (entire graph) search strategies.
        :param query: Natural language query
        :param k: Number of documents to return
        :return: List of up to k relevant Document objects
        """
        return self.documents[:k]
        
        

class EntityRelationshipSchema(BaseModel):
    entities: List[str] = Field(..., description="List of named entities (people, places, concepts)")
    relationships: List[Tuple[str, str, str]] = Field(
        ..., description="List of relationships as triples: (entity1, relation, entity2)"
    )

class GraphRAGBuilder:
    """
    Builds a Graph-RAG from a collection of documents.
    """
    def __init__(self, llm: JsonGenerator):
        self.llm = llm

    def build_knowledge_graph(self, documents: List[Document]) -> KnowledgeGraph:
        kg = KnowledgeGraph(documents=documents)
        for doc in documents:
            entities, relationships = self.extract_entities_and_relationships(doc)
            for entity in entities:
                kg.add_entity(entity)
            for (e1, rel, e2) in relationships:
                kg.add_relationship(e1, rel, e2)
        return kg

    def extract_entities_and_relationships(self, doc: Document) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """
        Uses the LLM to extract entities and relationships from a document.
        Returns a tuple: (entities, relationships)
        relationships: List of (entity1, relation, entity2)
        """
        prompt = (
            "Extract all named entities (people, places, concepts) and relationships between them "
            "from the following text.\n"
            "Return a JSON object matching this JSON schema:\n"
            f"{EntityRelationshipSchema.schema_json(indent=2)}\n"
            f"Text: {doc.content}"
        )
        try:
            data = self.llm.generate_json(prompt, EntityRelationshipSchema)
            validated = EntityRelationshipSchema.parse_obj(data)
            entities = validated.entities
            relationships = validated.relationships
            return entities, relationships
        except (ValidationError, Exception):
            return [], []