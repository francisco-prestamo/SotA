from typing import List

def build_feature_consolidation_prompt(
    feature_name: str,
    paper_title: str,
    feature_values: List[str]
) -> str:
    """Build prompt for consolidating feature values across chunks and experts"""
    values_list = "\n".join([f"- {value}" for value in feature_values if value and value != "Not Available"])
    
    if not values_list:
        return f"No valid values found for feature '{feature_name}' in paper '{paper_title}'. Respond with 'Not Available'."
    
    return f"""You need to consolidate multiple extracted values for the feature '{feature_name}' from the paper '{paper_title}'.

EXTRACTED VALUES:
{values_list}

INSTRUCTIONS:
1. Synthesize these values into a single, coherent summary
2. Preserve all important information while removing redundancy
3. If values conflict, prioritize the most specific and detailed information
4. Keep the consolidated value concise but comprehensive
5. Maintain technical accuracy

Provide only the consolidated value as a string response."""
