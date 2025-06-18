from pydantic import BaseModel, Field
from typing import List

class LocalSearchModel(BaseModel):
    """Model for local search results"""
    answer: str = Field(description="Detailed answer from local search")
    evidence: List[str] = Field(description="List of evidence sources supporting the answer")
    confidence_score: float = Field(description="Confidence score for this local search result")
    entity_mentions: List[str] = Field(description="Key entities mentioned in the search results")
