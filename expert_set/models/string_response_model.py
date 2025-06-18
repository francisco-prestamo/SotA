from pydantic import BaseModel, Field

class StringResponseModel(BaseModel):
    """Model for a simple string response from a JSON generator."""
    response: str = Field(description="The string response")
