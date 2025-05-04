from abc import ABC, abstractmethod
from entities.embedding import Embedding

class DocEmbedder(ABC):
    """
    Interface for generating vector embeddings from documents.
    """

    @abstractmethod
    def embed(self, document_text: str) -> Embedding:
        """
        Generate an embedding for the given document content.

        Args:
            document_text (str): Document text to be embedded.

        Returns:
            Sequence[float]: The vector embedding of the document.
        """
        ...

