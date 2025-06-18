from pydantic import BaseModel, Field
from typing import List

class FinalResponseModel(BaseModel):
    """Model for the final structured response"""
    executive_summary: str = Field(description="High-level executive summary of findings")
    global_insights: str = Field(description="Key insights from global community analysis")
    local_findings: List[str] = Field(description="Important findings from local searches")
    confidence_assessment: str = Field(description="Overall confidence assessment and limitations")
    recommendations: List[str] = Field(description="Recommendations for further exploration")
