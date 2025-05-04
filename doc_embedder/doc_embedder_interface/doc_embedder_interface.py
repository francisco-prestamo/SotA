from abc import ABC, abstractmethod
from typing import List
from doc_embedder.embedding_interface.embedding_interfaces import EmbeddingInterface

class DocEmbedderInterface(ABC):
    """
    Interface for generating vector embeddings from documents.
    """

    @abstractmethod
    def embed(self, document_text: str) -> EmbeddingInterface:
        """
        Generate an embedding for the given document content.

        Args:
            document_text (str): Document text to be embedded.

        Returns:
            Sequence[float]: The vector embedding of the document.
        """
        ...