from pydantic import BaseModel
from typing import List, Optional
from graphrag.models.graph_types import Entity, Relationship

class SummaryCommunityModel(BaseModel):
    key_entities: List[Entity]
    key_relationships: List[Relationship]
    summary: str
