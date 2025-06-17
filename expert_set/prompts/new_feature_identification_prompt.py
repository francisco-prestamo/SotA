from typing import List

def build_new_feature_identification_prompt(
    expert_name: str,
    expert_description: str,
    document_title: str,
    document_authors: List[str],
    chunk_content: str,
    existing_features: List[str]
) -> str:
    """Build prompt for identifying new features in a document chunk"""
    existing_features_list = "\n".join([f"- {feature}" for feature in existing_features])
    authors_str = ", ".join(document_authors)
    
    return f"""You are {expert_name}, {expert_description}.

Your task is to identify NEW features in the given document chunk that are NOT already covered by the existing features.

EXISTING FEATURES (DO NOT include these):
{existing_features_list}

DOCUMENT INFORMATION:
- Title: {document_title}
- Authors: {authors_str}

DOCUMENT CHUNK:
{chunk_content}

INSTRUCTIONS:
1. Look for important characteristics, metrics, methods, or properties mentioned in the chunk
2. Only identify features that are NOT already covered by the existing features list
3. Focus on features that would be valuable for comparing papers in a SOTA table
4. Provide both the feature name and its value from this chunk
5. Limit to the 3 most important new features

Respond as a JSON object with:
- "new_features": array of new feature names
- "feature_values": object mapping each new feature name to its value in this chunk"""
