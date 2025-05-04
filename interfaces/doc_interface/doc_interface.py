from abc import ABC, abstractmethod
from typing import Any, Mapping, List

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
    def embedding(self) -> List[float]:
        """Vector embedding of the document."""
        ...

