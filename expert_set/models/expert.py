from pydantic import BaseModel


from ..models import DocumentChunk
from ..interfaces import KnowledgeRepository


class ExpertDescription(BaseModel):
    name: str
    description: str


class Expert:
    expert_model: ExpertDescription
    knowledge: KnowledgeRepository[DocumentChunk]

    def __init__(
        self, name: str, description: str, knowledge: KnowledgeRepository
    ) -> None:
        self.name = name
        self.expert_model = ExpertDescription.model_construct()
        self.expert_model.description = description
        self.expert_model.name = name
        self.knowledge = knowledge
