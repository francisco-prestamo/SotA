from typing import List, Tuple

from pydantic import ValidationError
from entities.document import Document
from graphrag.interfaces.json_generator import JsonGenerator
from graphrag.knowledge_graph import KnowledgeGraph
from graphrag.prompts.extract_graph import EntityRelationshipSchema, extract_graph_prompt
from graphrag.text_chunking import chunk_text


class GraphRAGBuilder:
    """
    Builds a Graph-RAG from a collection of documents.
    """
    def __init__(self, llm: JsonGenerator):
        self.llm = llm

    def build_knowledge_graph(self, documents: List[Document]) -> KnowledgeGraph:
        kg = KnowledgeGraph(documents=documents)
        for doc in documents:
            entities, relationships = self.extract_entities_and_relationships_from_doc(doc)
            for entity in entities:
                kg.add_entity(entity.lower())
            for (e1, rel, e2) in relationships:
                kg.add_relationship(e1.lower(), rel.lower(), e2.lower())
        return kg

    def extract_entities_and_relationships_from_doc(self, doc: Document) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """
        Uses the LLM to extract entities and relationships from a document.
        Returns a tuple: (entities, relationships)
        relationships: List of (entity1, relation, entity2)
        """
        chunks = []
        if doc.content:
            chunks = chunk_text(doc.content)
        else:
            chunks = chunk_text(doc.abstract)

        entities = []
        relationships = []

        for chunk in chunks:
            current_entities, current_relationships = self.extract_entities_and_relationships_from_text(chunk)
            entities += current_entities
            relationships += current_relationships

        return entities, relationships



    def extract_entities_and_relationships_from_text(self, text: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        try:
            data = self.llm.generate_json(extract_graph_prompt(text), EntityRelationshipSchema)
            validated = EntityRelationshipSchema.model_validate(data)
            entities = validated.entities
            relationships = validated.relationships
            return entities, relationships
        except (ValidationError, Exception):
            print("error in graph extraction")
            return [], []
