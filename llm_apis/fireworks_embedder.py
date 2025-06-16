from typing import Optional
import numpy as np
from entities.embedding import Embedding
from rag_repo.interfaces.text_embedder import RagRepoTextEmbedder
import os
import requests
from dotenv import load_dotenv

class FireworksEmbedding(Embedding):
    def similarity(self, other: Embedding) -> Optional[float]:
        if self.vector.shape != other.vector.shape:
            return None
        norm_self = np.linalg.norm(self.vector)
        norm_other = np.linalg.norm(other.vector)
        if norm_self == 0 or norm_other == 0:
            return None
        return float(np.dot(self.vector, other.vector) / (norm_self * norm_other))


FIREWORKS_API_URL = "https://api.fireworks.ai/inference/v1/embeddings"
load_dotenv()

class FireworksEmbedder(RagRepoTextEmbedder):
    def __init__(self, model: str = "nomic-ai/nomic-embed-text-v1.5", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.getenv("FIREWORKS_API_KEY")
        if not self.api_key:
            raise ValueError("Fireworks API key must be provided via argument or FIREWORKS_API_KEY env variable.")
        self._dim = self._probe_dim()

    def embed(self, text: str) -> Embedding:
        payload = {
            "model": self.model,
            "input": [text]
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        print("embedder querying...")
        response = requests.post(FIREWORKS_API_URL, json=payload, headers=headers)
        print("answer: ", response)
        response.raise_for_status()

        result = response.json()
        vector = np.array(result["data"][0]["embedding"], dtype=np.float32)
        return FireworksEmbedding(vector)

    @property
    def dim(self) -> int:
        return self._dim

    def _probe_dim(self) -> int:
        dummy_embedding = self.embed("test")
        return dummy_embedding.vector.shape[0]
