from pydantic import BaseModel, Field
from typing import List

class NewFeaturesListModel(BaseModel):
    reasoning: str = Field(description="Reasoning of why you should or shouldn't add new features to the feature set for future extraction")
    new_features: List[str]
