from pydantic import BaseModel

class SimpleTextAnswerModel(BaseModel):
    content: str

