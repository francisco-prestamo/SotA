from abc import ABC, abstractmethod
from typing import Set
from doc_database.doc_interface import Document
from doc_embedder.doc_embedder_interface.doc_embedder_interface import DocEmbedderInterface

class DocRecoverer(ABC):
    """Interface for recovering documents based on a text query."""

    def __init__(self,embedder: DocEmbedderInterface):
        """
        Initializes the DocRecoverer with a document embedder.
        """
        self.embedder = embedder

    @abstractmethod
    def recover(self, query: str) -> Set[Document]:
        """
        Search logic to retrieve documents matching the given query.

        Args:
            query: A text query.

        Returns:
            A set of Document instances matching the query.
        """
        ...