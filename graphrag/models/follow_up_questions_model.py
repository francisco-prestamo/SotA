from pydantic import BaseModel, Field
from typing import List

class FollowUpQuestionsModel(BaseModel):
    """Model for generating structured follow-up questions"""
    questions: List[str] = Field(description="List of relevant follow-up questions to explore")
    question_types: List[str] = Field(description="Types of each question (entity, relationship, temporal, causal)")
    priority_scores: List[float] = Field(description="Priority scores for each question (0.0-1.0)")
