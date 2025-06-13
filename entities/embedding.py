from abc import ABC, abstractmethod
from typing import List, Optional, Self
import numpy as np

class Embedding:
    """
    Interface for representing a vector embedding.
    """
    def __init__(self, vector: np.ndarray):
        self._vector = vector

    @property
    def vector(self) -> np.ndarray:
        """
        :return: The vector representation of the embedding
        """
        return self._vector

    @abstractmethod
    def similarity(self, other: Self) -> Optional[float]:
        """
        :return: The similarity with another embedding if this operation is valid
        """
