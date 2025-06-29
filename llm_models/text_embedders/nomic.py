import openai
from graphrag.interfaces.text_embedder import TextEmbedder
from rag_repo.interfaces import RagRepoTextEmbedder
from entities.embedding import Embedding
import time
import numpy as np


class NomicAIEmbedder(TextEmbedder, RagRepoTextEmbedder):
    def __init__(self, dimensions: int = 128):
        self.api_keys = ["fw_3ZNnU48srVX34yNW6P4SoZjL", "fw_3ZghXR53MQMWFzcCYBWWLSa9"]
        self.base_url = "https://api.fireworks.ai/inference/v1"
        self.client = None
        self.current_key_index = 0
        self.dimensions = dimensions

    def _set_client(self, api_key):
        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key=api_key,
        )

    @property
    def dim(self) -> int:
        return self.dimensions

    def embed(self, text: str):
        i = 0
        while True:
            api_key = self.api_keys[self.current_key_index]
            self._set_client(api_key)
            try:
                response = self.client.embeddings.create(
                    model="nomic-ai/nomic-embed-text-v1.5",
                    input=text,
                    dimensions=self.dimensions,
                )
                return Embedding(vector=np.array(response.data[0].embedding))
            except Exception as e:
                time.sleep(2)
                self.current_key_index = (self.current_key_index + 1) % len(
                    self.api_keys
                )
                i += 1
                if i >= 80:
                    raise RuntimeError(
                        "All API keys failed. Please check your API keys and network connection."
                    )
