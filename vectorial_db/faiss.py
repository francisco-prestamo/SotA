from typing import List
import faiss
import numpy as np

from rag_repo.interfaces.vectorial_db import VectorialDB


class FaissVectorialDB(VectorialDB):
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.id_map = faiss.IndexIDMap(self.index)
        self.next_id = 0

    def store(self, vector: np.ndarray) -> int:
        if vector.ndim != 1 or vector.shape[0] != self.dim:
            raise ValueError(f"Expected 1D vector of dimension {self.dim}, got shape {vector.shape}")
        vec = np.expand_dims(vector.astype(np.float32), axis=0)
        vec_id = self.next_id
        self.id_map.add_with_ids(vec, np.array([vec_id], dtype=np.int64))
        self.next_id += 1
        return vec_id

    def get_closest(self, vector: np.ndarray, k: int) -> List[int]:
        if vector.ndim != 1 or vector.shape[0] != self.dim:
            raise ValueError(f"Expected 1D vector of dimension {self.dim}, got shape {vector.shape}")
        if self.id_map.ntotal == 0:
            return []
        vec = np.expand_dims(vector.astype(np.float32), axis=0)
        _, indices = self.id_map.search(vec, k)
        return indices[0].tolist()
