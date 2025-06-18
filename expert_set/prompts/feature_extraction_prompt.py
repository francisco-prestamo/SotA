from typing import List

def build_feature_extraction_prompt(
    expert_name: str,
    expert_description: str,
    document_title: str,
    document_authors: List[str],
    chunk_content: str,
    features_to_extract: List[str]
) -> str:
    """Build prompt for extracting specific features from a document chunk"""
    features_list = "\n".join([f"- {feature}" for feature in features_to_extract])
    authors_str = ", ".join(document_authors)
    
    return f"""You are {expert_name}, {expert_description}.

Your task is to extract specific features from the given document chunk. You must provide a value for each feature listed below.

FEATURES TO EXTRACT:
{features_list}

DOCUMENT INFORMATION:
- Title: {document_title}
- Authors: {authors_str}

DOCUMENT CHUNK:
{chunk_content}

INSTRUCTIONS:
1. For each feature listed above, extract the most relevant information from the chunk
2. If a feature is not explicitly mentioned or cannot be inferred, respond with "Not Available"
3. Provide concise but informative values
4. Focus on factual information from the text

Respond as a JSON object with feature names as keys and extracted values as the response."""
