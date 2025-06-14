<<<<<<< Updated upstream
from abc import ABC, abstractmethod
from typing import Optional
from entities.embedding import Embedding
from pydantic import BaseModel, Field

class Document(BaseModel):
    id: str
    title: str
    abstract: str
    content: str
=======
from pydantic import BaseModel

class Document(BaseModel):
    """Represents a recoverable document."""

    id: str
    """Unique identifier of the document."""

    title: str
    """Title of the document."""

    abstract: str
    """Abstract of the document."""

    authors: list[str]
    """List of authors of the document."""

    content: str
    """Contents of the document."""

    def __eq__(self, other):
        if not isinstance(other, Document):
            return False
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)
>>>>>>> Stashed changes
