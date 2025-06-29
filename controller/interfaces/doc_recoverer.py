from abc import ABC, abstractmethod
from typing import Set, Optional, Tuple
from entities.document import Document

class DocRecoverer(ABC):
    """Interface for recovering documents based on a text query."""

    @property
    def name(self) -> str:
        """
        Name of the document recoverer.

        Returns:
            A string representing the name of the recoverer.
        """
        return "Document Recoverer"

    @property
    def description(self) -> str:
        """
        Description of the document recoverer.

        Returns:
            A string describing the recoverer.
        """
        return "A document recoverer that retrieves documents based on a text query."

    @abstractmethod
    def recover(self, query: str, k: int, date_filter: Optional[Tuple[str, str]] = None) -> Set[Document]:
        """
        Search logic to retrieve documents matching the given query.

        Args:
            query: A text query.
            k: Number of documents to retrieve
            date_filter: Optional tuple of start and end dates for filtering documents by publication date.

        Returns:
            A set of Document instances matching the query.
        """
        ...