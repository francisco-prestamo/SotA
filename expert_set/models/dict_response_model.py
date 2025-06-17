from pydantic import BaseModel, Field
from typing import Dict, Any

class DictResponseModel(BaseModel):
    """Model for a dictionary response from a JSON generator."""
    data: Dict[str, Any] = Field(description="The dictionary response")
