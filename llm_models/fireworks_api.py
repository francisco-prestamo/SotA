import os
from typing import Optional, Type, TypeVar
import fireworks.client
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
import numpy as np
import time

from entities.embedding import Embedding

import config
from receptionist_agent.interfaces import JsonGenerator as ReceptionistJsonGen
from expert_set.interfaces import JsonGenerator as ExpertSetJsonGen
from recoverer_agent.interfaces import JsonGenerator as RecovererJsonGen
from rag_repo.interfaces import RagRepoTextEmbedder
from graphrag.interfaces import (
    JsonGenerator as JsonGenerator,
    TextEmbedder as GraphRagTextEmbedder,
)
from board.interfaces import JsonGenerator as BoardJsonGen

load_dotenv()
T = TypeVar("T", bound=BaseModel)

models = [
    "accounts/fireworks/models/llama4-scout-instruct-basic",  # $0.15 input, $0.60 output
    "accounts/fireworks/models/qwen3-30b-a3b",  # $0.22 input, $0.60 output
    "accounts/fireworks/models/qwen3-235b-a22b",  # $0.22 input, $0.88 output
    "accounts/fireworks/models/deepseek-r1",  # $3.00 input, $8.00 output
]


class FireworksEmbedding(Embedding):
    def similarity(self, other: Embedding) -> Optional[float]:
        if self.vector.shape != other.vector.shape:
            return None
        norm_self = np.linalg.norm(self.vector)
        norm_other = np.linalg.norm(other.vector)
        if norm_self == 0 or norm_other == 0:
            return None
        return float(np.dot(self.vector, other.vector) / (norm_self * norm_other))


class FireworksApi(
    ReceptionistJsonGen,
    ExpertSetJsonGen,
    RecovererJsonGen,
    JsonGenerator,
    RagRepoTextEmbedder,
    GraphRagTextEmbedder,
    BoardJsonGen,
):
    """Implementation of various JSON generation interfaces using the Fireworks API."""

    def __init__(self):

        fireworks_api_key = os.getenv("FIREWORKS_API_KEY")
        self._dim = None

        if not fireworks_api_key:
            raise ValueError(
                "No API key found in the .env file. Please add your FIREWORKS_API_KEY to the .env file."
            )

        fireworks.client.api_key = fireworks_api_key
        self.fireworks_api_key = fireworks_api_key

    def generate_json(self, query: str, schema: Type[T]) -> T:
        if config.inspect_query():
            print("=" * 60)
            print(query)
            input()

        client = fireworks.client.Fireworks(api_key=self.fireworks_api_key)
        response = client.chat.completions.create(
            model=models[0],
            response_format={
                "type": "json_object",
                "schema": schema.model_json_schema(),
            },
            messages=[
                {"role": "system", "content": query},
            ],
            temperature=0.2,
        )
        print("Waiting 6 seconds...")
        time.sleep(6)
        print("Wait is up")
        response = response.choices[0].message.content
        if config.inspect_query():
            print("-" * 60)
            print(response)
            print("=" * 60)
            input()
        response = schema.model_validate_json(response)
        return response

    def embed(self, text: str) -> Embedding:
        return self.embed_texts([text])[0]

    @property
    def dim(self) -> int:
        if not self._dim:
            self._dim = len(self.embed("hello hello").vector)

        return self._dim

    def embed_texts(self, texts: list[str]) -> list[Embedding]:
        client = OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=self.fireworks_api_key,
        )

        answ = []
        for text in texts:
            vector = (
                client.embeddings.create(
                    model="nomic-ai/nomic-embed-text-v1.5", input=text
                )
                .data[0]
                .embedding
            )
            print("Waiting 6 seconds...")
            time.sleep(6)
            print("Wait is up")
            embedding = FireworksEmbedding(np.array(vector))
            answ += [embedding]

        return answ
