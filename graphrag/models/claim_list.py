from typing import List
from pydantic import BaseModel
from graphrag.models.graph_types import Claim

class ClaimListModel(BaseModel):
    claims: List[Claim]
