from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class ScrapperQuery(BaseModel):
    """Model for a scrapper query"""
    query: str = Field(..., description="The search query to use")
    reasoning: str = Field(..., description="Reasoning for this specific query")

class ScrapperSelection(BaseModel):
    """Model for selecting scrapers and their queries"""
    source_name: str = Field(..., description="Name of the selected source/scrapper")
    selected: bool = Field(..., description="Whether this source is selected")
    queries: List[ScrapperQuery] = Field(
        default_factory=list, 
        description="List of queries to use with this scrapper (up to 3)"
    )
    source_reasoning: str = Field(..., description="Reasoning for selecting or not selecting this source")

class ScrapperSelectionResponse(BaseModel):
    """Model for the complete scrapper selection response"""
    reasoning: str = Field(..., description="Overall reasoning for the selections made")
    selections: List[ScrapperSelection] = Field(..., description="Selected scrapers with their queries")
