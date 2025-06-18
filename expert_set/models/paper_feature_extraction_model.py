from pydantic import BaseModel
from typing import List, Dict
from entities.document import Document
from expert_set.models.expert_chunk_new_features_model import ExpertChunkNewFeatures

class PaperFeatureExtraction(BaseModel):
    """Complete feature extraction results for a paper"""
    document: Document
    chunk_features: List[Dict[str, str]]  # List of feature dictionaries from each expert/chunk
    chunk_new_features: List[ExpertChunkNewFeatures]
    consolidated_features: Dict[str, str]
    consolidated_new_features: Dict[str, str]
