import spacy
from typing import List

# Load spaCy with only the tokenizer enabled
nlp = spacy.load(
    "en_core_web_sm",
    disable=["parser", "tagger", "ner", "lemmatizer", "attribute_ruler"],
)


def chunk_text(text: str, window_size: int = 3000, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping token chunks.

    Args:
        text (str): The input text to be chunked.
        window_size (int): Number of tokens per chunk.
        overlap (int): Number of overlapping tokens between chunks.

    Returns:
        List[str]: A list of chunked text strings.
    """
    if window_size <= 0:
        raise ValueError("window_size must be greater than 0")
    if overlap < 0 or overlap >= window_size:
        raise ValueError("overlap must be in range [0, window_size)")

    doc = nlp(text)
    tokens = [token.text for token in doc]
    chunks = []

    step = window_size - overlap
    for start in range(0, len(tokens), step):
        end = start + window_size
        chunk = tokens[start:end]
        if chunk:
            chunks.append(" ".join(chunk))

        if end >= len(tokens):
            break

    return chunks
