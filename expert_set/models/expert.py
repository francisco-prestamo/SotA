from pydantic import BaseModel
from ..interfaces import KnowledgeRepository

class ExpertModel(BaseModel):
    name: str 
    description: str 

class Expert:
    expert_model: ExpertModel
    knowledge: KnowledgeRepository

    def __init__(self, name: str, description: str, knowledge: KnowledgeRepository) -> None:
        self.name = name
        self.expert_model = ExpertModel.model_construct()
        self.expert_model.description = description
        self.expert_model.name = name
        self.knowledge = knowledge
            
