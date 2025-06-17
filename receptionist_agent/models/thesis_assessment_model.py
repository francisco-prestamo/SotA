from pydantic import BaseModel, Field
from typing import List, Optional


class ThesisAssessmentModel(BaseModel):
    """Model for assessing if there is sufficient knowledge about a thesis topic"""

    reasoning: str = Field(description="Reasoning behind the assessment")
    is_sufficient: bool = Field(
        description="Whether the current knowledge about the thesis topic is sufficient"
    )
    missing_aspects: List[str] = Field(
        description="List of aspects that are missing in the current knowledge"
    )
    suggested_questions: List[str] = Field(
        description="Suggested questions to ask to gather more knowledge", default=[]
    )
