from pydantic import BaseModel, Field
from typing import List

class ExpertModel(BaseModel):
    """Model for expert description"""
    name: str = Field(description="Name or identifier of the expert")
    description: str = Field(description="Brief description of the expert's specialties and background")
    query: str = Field(description="Query to search for surveys about the expert's topic")
