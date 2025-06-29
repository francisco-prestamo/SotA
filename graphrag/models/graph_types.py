from typing import List, Optional, Tuple
from enum import Enum
from pydantic import BaseModel
from entities.embedding import Embedding


class EntityType(str, Enum):
    PERSON = "Person"
    ORGANIZATION = "Organization"
    LOCATION = "Location"
    EVENT = "Event"
    CONCEPT = "Concept"
    DATE = "Date"
    TIME = "Time"
    OTHER = "Other"

class Entity(BaseModel):
    name: str
    type: EntityType
    description: str

class Relationship(BaseModel):
    description: str
    source: str
    target: str


class Claim(BaseModel):
    subject: str
    object: str
    claim_type: str
    claim_status: str
    claim_description: str
    claim_date: dict
    claim_source_text: list[str]

class CommunityReport(BaseModel):
    key_entities: List[Entity]
    key_relationships: List[Relationship]
    summary: str
    embedding: Embedding

    class Config:
        arbitrary_types_allowed = True

class Community(BaseModel):
    id: str
    level: int
    members: List[Tuple[str, EntityType]]
    parent: Optional[str]
    report: Optional[CommunityReport]

class TextUnit(BaseModel):
    document_id: str
    text: str
    unit_id: Optional[str]
    position: Optional[int]
