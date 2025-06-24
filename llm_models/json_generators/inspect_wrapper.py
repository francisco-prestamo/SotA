from typing import Type, TypeVar

from pydantic import BaseModel
from graphrag.interfaces.json_generator import JsonGenerator as GraphRagJsonGen, T
from board.board import JsonGenerator as BoardJsonGen
from expert_set.interfaces import JsonGenerator as ExpertSetJsonGen
from recoverer_agent.interfaces import JsonGenerator as RecovJsonGen
from receptionist_agent.interfaces import JsonGenerator as ReceptJsonGen
from mocks.user_agent.interfaces import JsonGenerator as UserAgentJsonGen
import json

T = TypeVar("T", bound=BaseModel)


class JsonGeneratorInspectionWrapper(
    GraphRagJsonGen,
    BoardJsonGen,
    ExpertSetJsonGen,
    RecovJsonGen,
    ReceptJsonGen,
    UserAgentJsonGen,
):
    def __init__(
        self,
        json_gen: (
            GraphRagJsonGen
            | BoardJsonGen
            | ExpertSetJsonGen
            | RecovJsonGen
            | ReceptJsonGen
            | UserAgentJsonGen
        ),
    ) -> None:
        self.json_gen = json_gen

    def generate_json(self, query: str, schema: Type[T]) -> T:

        print("Generating json...")

        print("Query: ")
        print("=" * 60)
        print(query)
        input()
        print("=" * 60)

        print("Schema: ")
        print("=" * 60)
        print(json.dumps(schema.model_json_schema(), indent=2))
        input()
        print("=" * 60)

        answer = self.json_gen.generate_json(query, schema)

        print("Answer: ")
        print("=" * 60)
        print(answer.model_dump_json(indent=2))
        input()
        print("=" * 60)

        return answer
