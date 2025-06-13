from abc import ABC, abstractmethod
from typing import List
from entities.embedding import Embedding

class TextEmbedder(ABC):
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
