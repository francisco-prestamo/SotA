from typing import List
from pydantic import BaseModel


class ChatEntry(BaseModel):
    role: str
    content: str

class Chat(BaseModel):
    entries: List[ChatEntry]
