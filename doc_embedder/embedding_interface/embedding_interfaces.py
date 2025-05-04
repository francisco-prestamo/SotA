from abc import ABC
from typing import List

class EmbeddingInterface(ABC):
    """
    Interface for representing a vector embedding.
    """

    def __init__(self, embedding: List[float]) -> None:
        """
        :param embedding: A list of floats representing the embedding.
        """
        self.embedding = embedding

    @property
    def embedding_dim(self) -> int:
        """
        :return: Dimensionality of the embedding vector.
        """
        return len(self.embedding)
