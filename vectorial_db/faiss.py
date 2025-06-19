from typing import List
import faiss
import numpy as np

from rag_repo.interfaces.vectorial_db import VectorialDB, VectorialDBFactory


class FaissVectorialDB(VectorialDB):
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.next_id = 0

    def store(self, vector: np.ndarray) -> int:
        if vector.ndim != 1 or vector.shape[0] != self.dim:
            raise ValueError(f"Expected 1D vector of dimension {self.dim}, got shape {vector.shape}")
        vec_id = self.next_id
        self.next_id += 1
        self.index.add(np.array([vector]))
        return vec_id

    def get_closest(self, vector: np.ndarray, k: int) -> List[int]:
        if vector.ndim != 1 or vector.shape[0] != self.dim:
            raise ValueError(f"Expected 1D vector of dimension {self.dim}, got shape {vector.shape}")
        _, indices = self.index.search(np.array([vector]), k)
        indices = indices[0].tolist()
        return [i for i in indices if i != -1]

class FaissVecDBFactory(VectorialDBFactory):
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def create_vectorial_db(self) -> VectorialDB:
        return FaissVectorialDB(self.dim)
