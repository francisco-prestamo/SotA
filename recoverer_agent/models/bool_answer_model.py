from pydantic import BaseModel

class BoolAnswerModel(BaseModel):
    reasoning: str
    answer: bool
