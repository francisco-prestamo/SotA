from typing import Dict, List, Tuple
from pydantic import BaseModel, create_model

class ExpertSearchReasoningModel(BaseModel):
    reasoning: str
    what_to_search: str

# This function dynamically builds a model for all experts at runtime
def build_expert_search_reasoning_model(expert_names: List[str]):
    fields = {
        expert: (ExpertSearchReasoningModel, ...)
        for expert in expert_names
    }
    return create_model(
        'DynamicExpertSearchReasoningModel',
        **fields
    )

# Example usage:
# expert_names = ['expert_1', 'expert_2']
# DynamicModel = build_expert_search_reasoning_model(expert_names)
# instance = DynamicModel(expert_1=ExpertSearchReasoningModel(reasoning='...', what_to_search='...'), ...)
