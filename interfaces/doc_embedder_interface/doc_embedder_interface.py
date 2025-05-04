from abc import ABC, abstractmethod
from typing import Sequence
from interfaces.doc_interface.doc_interface import Document
from interfaces.doc_database_interface.doc_database_interface import DocDatabaseInterface

class DocEmbedderInterface(ABC):
    """
    Interface for generating vector embeddings from documents.
    """
    def __init__(self, doc_database: DocDatabaseInterface):
        """
        :param doc_database: An implementation of DocDatabaseInterface for storing and retrieving embeddings.
        """
        self.doc_database = doc_database

    @abstractmethod
    def embed_by_id(self, document_id: str) -> Sequence[float]:
        """
        Generate an embedding for a document identified by its ID.

        Args:
            document_id (str): The unique identifier of the document.

        Returns:
            Sequence[float]: The vector embedding of the document.
        """
        ...

    @abstractmethod
    def embed(self, document: Document) -> Sequence[float]:
        """
        Generate an embedding for the given document content.

        Args:
            document (str): The raw content of the document.

        Returns:
            Sequence[float]: The vector embedding of the document.
        """
        ...