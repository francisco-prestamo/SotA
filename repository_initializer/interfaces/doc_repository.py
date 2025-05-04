from abc import ABC, abstractmethod

from entities.document import Document

class DocRepository(ABC):

    @abstractmethod
    def insert(self, collection: str, document: Document) -> str:
        """
        Insert a document into the given collection.
        Returns the new document's ID.
        """
        ...
