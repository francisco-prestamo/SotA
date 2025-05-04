from abc import ABC, abstractmethod
from typing import Any, Mapping, List
from doc_embedder.embedding_interface.embedding_interfaces import EmbeddingInterface

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
    def embedding(self) -> EmbeddingInterface:
        """Vector embedding of the document."""
        ...

