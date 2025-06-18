from pydantic import BaseModel


class DocumentChunk(BaseModel):
    document_id: str
    document_title: str
    chunk: str
