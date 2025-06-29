import spacy
from typing import List, Optional
from entities.document import Document
from graphrag.models.text_unit import TextUnit
from pydantic import ValidationError
from graphrag.interfaces.text_embedder import TextEmbedder
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load spaCy with only the tokenizer enabled
nlp = spacy.load(
    "en_core_web_sm",
)

def chunk_text(text: str, max_tokens=3000, overlap_tokens=50) -> list[str]:
    """
    Chunk text into semantically meaningful chunks using spaCy, with overlap.

    Args:
        text (str): Input text to be chunked.
        max_tokens (int): Maximum number of tokens per chunk.
        overlap_tokens (int): Number of tokens to overlap between chunks.

    Returns:
        List[str]: list of chunks.
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    chunks = []
    current_chunk = []
    current_len = 0
    i = 0

    while i < len(sentences):
        sentence = sentences[i]
        sentence_doc = nlp(sentence)
        sent_len = len(sentence_doc)

        if current_len + sent_len <= max_tokens:
            current_chunk.append(sentence)
            current_len += sent_len
            i += 1
        else:
            # Save current chunk
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)

            # Prepare for next chunk with overlap
            overlap_chunk = []
            overlap_len = 0
            for sent in reversed(current_chunk):
                sent_len = len(nlp(sent))
                if overlap_len + sent_len <= overlap_tokens:
                    overlap_chunk.insert(0, sent)
                    overlap_len += sent_len
                else:
                    break

            current_chunk = overlap_chunk
            current_len = overlap_len

    # Add remaining chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append(chunk_text)

    return chunks

def chunk_document(text_embedder: TextEmbedder, document: Document, max_tokens=30000, overlap_tokens=50) -> List[TextUnit]:
    """
    Chunk document's content into semantically meaningful chunks using spaCy, with overlap,
    and convert them to TextUnit objects.

    Args:
        document (Document): Input document to be chunked.
        max_tokens (int): Maximum number of tokens per chunk.
        overlap_tokens (int): Number of tokens to overlap between chunks.

    Returns:
        List[TextUnit]: list of TextUnit objects created from the chunks.
    """
    if not document.content:
        raise ValidationError("Document content is empty")
    
    # Get raw text chunks using the chunking function
    text_chunks = chunk_text(document.content, max_tokens=max_tokens, overlap_tokens=overlap_tokens)

    def create_text_unit(i_chunk):
        i, chunk = i_chunk
        return TextUnit(
            document_id=document.id,
            text=chunk,
            unit_id=f"{document.id}_chunk_{i}",
            position=i,
            number_tokens=len(nlp(chunk)),
            embedding=text_embedder.embed(chunk)
        )

    text_units = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(create_text_unit, (i, chunk)): i for i, chunk in enumerate(text_chunks)}
        for future in as_completed(futures):
            text_unit = future.result()
            text_units.append(text_unit)

    # Sort to preserve original order
    text_units.sort(key=lambda tu: tu.position)
    return text_units
