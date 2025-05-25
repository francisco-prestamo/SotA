import os
from typing import Type, TypeVar
from fireworks.client import Fireworks
from pydantic import BaseModel, ValidationError
from graphrag.interfaces.json_generator import JsonGenerator
from rar_engine.interfaces.text_text_llm import TextGenerator

T = TypeVar("T", bound=BaseModel)

class FireworksApi(JsonGenerator, TextGenerator):
    def __init__(self, api_key: str, model: str = "accounts/fireworks/models/deepseek-v3"):
        self.model = model
        self.client = Fireworks(api_key=api_key)

    def generate_json(self, query: str, schema: Type[T]) -> T:
        # Call Fireworks completion API
        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object", "schema": schema.model_json_schema()},
            messages=[
                {"role": "user", "content": query}
            ],
        )

        # Extract and parse response
        content = response.choices[0].message.content.strip()

        try:
            json_data = schema.parse_raw(content)
            return json_data
        except ValidationError as ve:
            raise ValueError(f"Model output could not be validated: {ve}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error parsing model output: {e}")

    def generate_text(self, query: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": query}
            ],
        )

        return response.choices[0].message.content.strip()




