from pydantic import BaseModel, Field
from typing import List

class InitialAnswerModel(BaseModel):
    """Model for structured initial answer generation"""
    answer: str = Field(description="Comprehensive initial answer based on community insights")
    key_insights: List[str] = Field(description="List of key insights extracted from communities")
    confidence_score: float = Field(description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="Explanation of how the answer was derived")
