from pydantic import BaseModel, Field
from typing import List

from expert_set.models.build_expert_model import BuildExpertCommand

class BuildExpertCommandList(BaseModel):
    """Model for generating a list of experts"""
    experts: List[BuildExpertCommand] = Field(description="List of expert descriptions")
