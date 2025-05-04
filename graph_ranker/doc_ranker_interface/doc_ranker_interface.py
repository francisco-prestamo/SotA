from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
from doc_embedder.embedding_interface.embedding_interfaces import EmbeddingInterface

@dataclass(frozen=True)
class RankedDocument:
    """
    Represents a document embedding with its associated relevance score.
    """
    doc_id: str
    score: float

@dataclass(frozen=True)
class DocumentsToRank:
    doc_id: str
    embedding: EmbeddingInterface

class DocRankerInterface(ABC):
    """
    Interface for document rankers that utilize embeddings to rank documents
    by relevance to a query embedding.
    """

    @abstractmethod
    def rank(
        self,
        query_embedding: EmbeddingInterface,
        docs: List[DocumentsToRank],
    ) -> List[RankedDocument]:
        """
        Given a query embedding and a list of document embeddings,
        compute and return each document’s relevance score.

        :param query_embedding: Embedding of the query.
        :param docs: List of documents to rank with.
        :return: A list of RankedDocument instances (embedding + score),
                 sorted by descending score.
        """
        pass