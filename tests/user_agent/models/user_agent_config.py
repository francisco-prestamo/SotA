from pydantic import BaseModel


class UserAgentConfig(BaseModel):
    paper_description: str
    personality_description: str
