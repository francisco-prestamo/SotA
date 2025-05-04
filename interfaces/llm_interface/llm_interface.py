from abc import ABC, abstractmethod
from typing import List, Dict, Any

class LLM(ABC):
    """Abstract base class for LLM implementations."""

    @abstractmethod
    def generate_text(self, query: str) -> str:
        """
        Send a query to the LLM and get back a text response.
        :param query: The input prompt or question.
        :return: A textual response from the model.
        """
        pass

    @abstractmethod
    def generate_embedding(self, query: str) -> List[float]:
        """
        Send a query to the LLM and get back an embedding vector.
        :param query: The input prompt or text to embed.
        :return: A list of floats representing the embedding.
        """
        pass

    @abstractmethod
    def generate_json(self, query: str) -> Dict[str, Any]:
        """
        Send a query to the LLM and get back a JSON-formatted response.
        :param query: The input prompt expecting structured output.
        :return: A dict parsed from the JSON response.
        """
        pass