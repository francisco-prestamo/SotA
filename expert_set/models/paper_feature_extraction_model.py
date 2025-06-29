from pydantic import BaseModel
from typing import List, Dict
from entities.document import Document
from expert_set.models.expert_chunk_new_features_model import ExpertChunkNewFeatures

class PaperFeatureExtraction(BaseModel):
    """Complete feature extraction results for a paper"""
    document: Document
    old_features: Dict[str, str]
    new_features: Dict[str, str]
