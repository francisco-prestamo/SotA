from pydantic import BaseModel, Field
from typing import List
from receptionist_agent.models.expert_model import ExpertModel

class ExpertsListModel(BaseModel):
    """Model for generating a list of experts"""
    experts: List[ExpertModel] = Field(description="List of expert descriptions")
