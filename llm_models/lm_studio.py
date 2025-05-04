from doc_recoverers.interfaces.doc_embedder import DocEmbedder
from rar_engine.interfaces.text_text_llm import TextGenerator
import requests
from typing import List

class LMStudio_Embedder(DocEmbedder):
    def __init__(self, url ,model:str) -> None:
        self.model = model
        self.url = url
        self._ensure_embedding_support()
        
    def _ensure_embedding_support(self):
        resp = requests.get(f"{self.url.rstrip('/')}/models")
        resp.raise_for_status()
        supported_models = resp.json().get("models", [])
        if self.model not in supported_models:
            raise ValueError(
                f"Model '{self.model}' is not supported by LM Studio. "
                f"Available models: {supported_models}"
            )
        
    def embed(self, text: str) -> List[float]:
        """
        Calls the LM Studio embedding endpoint and returns a vector for the given text.
        """
        payload = {
            "model": self.model,
            "input": text
        }
        resp = requests.post(
            f"{self.url.rstrip('/')}/embeddings",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        resp.raise_for_status()
        data = resp.json()
        # LM Studio returns the embedding under the "embedding" key
        return data.get("embedding", [])




