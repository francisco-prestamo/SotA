from abc import ABC, abstractmethod
from entities.embedding import Embedding

class RagRepoTextEmbedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> Embedding:
        """
        Embed a single text into a vector representation.

        Args:
            text (str): Input text string to embed.

        Returns:
            Embedding: The embedding vector for the input text.
        """
        pass

    @property
    @abstractmethod
    def dim(self) -> int:
        pass
