from abc import ABC, abstractmethod
from typing import Optional
from entities.embedding import Embedding

class Document(ABC):
    """Represents a recoverable document."""

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier of the document."""
        ...

    @property
    @abstractmethod
    def abstract(self) -> str:
        """Abstract of the document."""
        ...

    @property
    @abstractmethod
    def content(self) -> Optional[str]:
        """Contents of the document, which may or may not be included"""

    @property
    @abstractmethod
    def embedding(self) -> Embedding:
        """Vector embedding of the document."""
        ...

