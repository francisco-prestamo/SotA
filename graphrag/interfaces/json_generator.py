from abc import ABC, abstractmethod
from typing import Any

class JsonGenerator(ABC):
    @abstractmethod
    def generate_json(self, query: str) -> Any:
        """
        Send a query to the LLM and get back a JSON response.
        :param query: The input prompt or question.
        :return: A JSON-compatible Python object (dict, list, etc.)
        """
        pass
