from abc import ABC, abstractmethod
from typing import List, Optional, Self

class Embedding(ABC):
    """
    Interface for representing a vector embedding.
    """

    @abstractmethod
    def similarity(self, other: Self) -> Optional[float]:
        """
        :return: The similarity with another embedding if this operation is valid
        """
