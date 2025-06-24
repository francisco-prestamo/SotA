from pydantic import BaseModel
from typing import List

class NewFeaturesListModel(BaseModel):
    new_features: List[str]
