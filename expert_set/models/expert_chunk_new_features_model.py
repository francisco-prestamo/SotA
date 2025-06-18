from pydantic import BaseModel, Field
from typing import List, Dict

class ExpertChunkNewFeatures(BaseModel):
    """New features identified by an expert in a specific chunk"""
    expert_name: str
    chunk_index: int
    new_features: List[str] = Field(description="List of newly identified feature names")
    new_feature_values: Dict[str, str] = Field(description="New feature name to value mapping")
