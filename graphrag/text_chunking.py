import spacy

nlp = spacy.load("en_core_web_sm")

def chunk_text(text: str, max_tokens=300, overlap_tokens=50) -> list[str]:
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
