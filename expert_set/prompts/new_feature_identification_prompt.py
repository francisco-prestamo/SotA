from typing import List


def build_new_feature_identification_prompt(
    expert_name: str,
    expert_description: str,
    document_title: str,
    document_authors: List[str],
    chunk_content: str,
    existing_features: List[str]
) -> str:
    """Build prompt for identifying NEW FEATURE NAMES ONLY in a document chunk."""
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


def build_feature_value_extraction_prompt(
        feature_name: str,
        document_title: str,
        document_authors: List[str],
        chunk_content: str
) -> str:
    """Build prompt for extracting the value of a specific feature from a document chunk."""
    authors_str = ", ".join(document_authors)

    return f"""Extract the value for the feature '{feature_name}' from the given document chunk.

DOCUMENT INFORMATION:
- Title: {document_title}
- Authors: {authors_str}

FEATURE TO EXTRACT: {feature_name}

DOCUMENT CHUNK:
{chunk_content}

INSTRUCTIONS:
1. Look for information related to '{feature_name}' in the chunk.
2. Provide a concise, specific value (e.g., "Yes"/"No", specific tool name, metric value, etc.).
3. If the feature is not mentioned or unclear, respond with "Not Available".
4. Keep the response brief and factual.

Examples of good responses:
- For "Tool Used": "GPT-4" or "ResNet-50" or "PyTorch"
- For "Pre-training Applied": "Yes" or "No"
- For "Dataset Used": "ImageNet" or "CIFAR-10"
- For "Evaluation Metric": "Accuracy" or "F1-score"

Respond with just the value as a simple string.
"""