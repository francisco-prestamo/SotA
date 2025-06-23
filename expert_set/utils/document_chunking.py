from typing import List
from entities.document import Document
from ..models import DocumentChunk
from . import chunk_text


def chunk_document(doc: Document,window_size = 3000) -> List[DocumentChunk]:
    chunks = chunk_text(doc.content,window_size=window_size)
    answ = []
    for chunk in chunks:
        answ.append(
            DocumentChunk(chunk=chunk, document_title=doc.title, document_id=doc.id)
        )

    return answ
