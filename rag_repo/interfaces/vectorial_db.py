from abc import ABC, abstractmethod
from typing import List
import numpy as np


class VectorialDB(ABC):
    @abstractmethod
    def store(self, vector: np.ndarray) -> int:
        pass

    @abstractmethod
    def get_closest(self, vector: np.ndarray, k: int) -> List[int]:
        pass
