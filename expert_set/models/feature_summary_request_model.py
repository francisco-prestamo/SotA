from pydantic import BaseModel
from typing import List

class FeatureSummaryRequest(BaseModel):
    """Request model for feature summarization"""
    feature_name: str
    paper_title: str
    feature_values: List[str]
