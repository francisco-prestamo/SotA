from abc import ABC, abstractmethod
from typing import Set
from interfaces.doc_interface.doc_interface import Document

class DocRecoverer(ABC):
    """Interface for recovering documents based on a text query."""

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