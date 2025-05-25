from typing import List, Tuple
from pydantic import BaseModel, Field

class EntityRelationshipSchema(BaseModel):
    entities: List[str] = Field(..., description="List of named entities (people, places, concepts)")
    relationships: List[Tuple[str, str, str]] = Field(
        ..., description="List of relationships as triples: (entity1, relation, entity2)"
    )

def extract_graph_prompt(text: str) -> str:
    return (
        "Extract all named entities (people, places, concepts) and relationships between them "
        "from the following text.\n"
        "Return a JSON object matching this JSON schema:\n"
        f"{EntityRelationshipSchema.model_json_schema()}\n"
        f"Text: {text}"
    )

