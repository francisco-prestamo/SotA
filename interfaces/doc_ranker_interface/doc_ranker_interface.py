from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
from interfaces.doc_database_interface.doc_database_interface import DocDatabaseInterface

@dataclass(frozen=True)
class RankedDocument:
    """
    Represents a document identifier with its associated relevance score.
    """
    doc_id: str
    score: float

class DocRankerInterface(ABC):
    """
    Interface for document rankers that utilize a vector database to retrieve
    and sort documents by relevance to a natural language query.
    """

    def __init__(self, doc_database: DocDatabaseInterface):
        """
        :param doc_database: An implementation of DocDatabaseInterface for vector searches.
        """
        self.doc_database = doc_database

    @abstractmethod
    def rank(self, query: str, top_k: int = 10) -> List[RankedDocument]:
        """
        Given a natural language query, retrieve and rank documents from the vector database.

        :param query: The search query.
        :param top_k: Maximum number of documents to return.
        :return: A list of RankedDocument instances sorted by descending relevance score.
        """
        pass