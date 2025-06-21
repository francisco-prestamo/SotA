from abc import ABC, abstractmethod
from typing import Type, TypeVar

from openai import BaseModel

T = TypeVar("T", bound=BaseModel)


class JsonGenerator(ABC):
    @abstractmethod
    def generate_json(self, query: str, schema: Type[T]) -> T:
        """
        Send a query to the LLM and get back a JSON response.
        :param query: The input prompt or question.
        :param schema: The Pydantic model class to enforce on the response.
        :return: An instance of the Pydantic model
        """
        pass
