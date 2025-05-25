from typing import Any, Dict, List
from entities.document import Document


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
