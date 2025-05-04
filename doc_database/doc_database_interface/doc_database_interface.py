from abc import ABC, abstractmethod
from typing import List, Optional
from doc_database.initial_topics_generator_interface.initial_topics_generator_interface import InitialTopicsGeneratorInterface
from doc_database.doc_interface.doc_interface import Document
class DocDatabaseInterface(ABC):
    """
    Abstract interface for a document‐oriented, vector‐enabled database.
    """
    
    @abstractmethod
    def initialize_topics(
        self, topics_generator: InitialTopicsGeneratorInterface
    ) -> None:
        """
        Initialize the database with necessary collections and indices
        using the provided InitialTopicsGeneratorInterface instance.
        """
        ...

    @abstractmethod
    def insert(self, collection: str, document: Document) -> str:
        """
        Insert a document into the given collection.
        Returns the new document's ID.
        """
        ...

    @abstractmethod
    def get(self, collection: str, doc_id: str) -> Optional[Document]:
        """
        Retrieve a document by its ID from the given collection.
        Returns None if not found.
        """
        ...

    @abstractmethod
    def update(
        self, collection: str, doc_id: str, updates: Document
    ) -> bool:
        """
        Update fields of a document identified by doc_id.
        Returns True if the update was acknowledged.
        """
        ...

    @abstractmethod
    def delete(self, collection: str, doc_id: str) -> bool:
        """
        Delete a document by its ID.
        Returns True if the deletion was acknowledged.
        """
        ...

    @abstractmethod
    def search(
        self, collection: str, filter_query
    ) -> List[Document]:
        """
        Find documents in a collection matching the filter_query.
        Returns a list of matching documents.
        """
        ...