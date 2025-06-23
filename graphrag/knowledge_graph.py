from typing import Any, Dict, List, Set, Tuple, Optional
from entities.document import Document
from graphrag.models.graph_types import Entity, Relationship, Claim, EntityType, Community, CommunityReport
from graphrag.models.text_unit import TextUnit
import random
from graphrag.utils.text_chunking import chunk_text
from pydantic import BaseModel, Field
from tqdm import tqdm

class RatedPoint(BaseModel):
    point: str = Field(description="An informative point extracted from the text")
    rating: int = Field(description="Importance rating from 1-10, with 10 being most important")

class IntermediateResponse(BaseModel):
    rated_points: List[RatedPoint]

class KnowledgeGraph:
    """
    In-memory knowledge graph for GraphRAG Knowledge Model.
    Stores Documents, TextUnits, Entities, Relationships, Covariates, Communities, and Community Reports.
    """
    def __init__(self, documents: List[Document]):
        self.documents: List[Document] = documents
        self.text_units: List[TextUnit] = []
        self.entities: List[Entity] = []
        self.relationships: List[Relationship] = []
        self.covariates: List[Claim] = []
        self.communities: List[Community] = []
        self.community_reports: List[CommunityReport] = []
        self.textunit_entities: Dict[str, List[Entity]] = {}

    def add_document(self, document: Document):
        self.documents.append(document)

    def add_text_unit(self, text_unit: TextUnit):
        self.text_units.append(text_unit)

    def add_entity(self, entity: Entity):
        self.entities.append(entity)

    def add_relationship(self, relationship: Relationship):
        self.relationships.append(relationship)

    def add_covariate(self, covariate: Claim):
        self.covariates.append(covariate)

    def add_community(self, community: Community):
        self.communities.append(community)

    def add_community_report(self, report: CommunityReport):
        self.community_reports.append(report)

    def add_textunits_entities(self, textunit_id: str, entities: List[Entity]):
        self.textunit_entities[textunit_id] = entities
