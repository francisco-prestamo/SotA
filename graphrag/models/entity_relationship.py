from typing import List
from pydantic import BaseModel, Field
from graphrag.models.graph_types import Entity, Relationship

class EntityRelationshipModel(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]
