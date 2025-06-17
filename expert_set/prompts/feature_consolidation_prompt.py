from typing import List, Optional

def build_feature_consolidation_prompt(
    feature_name: str,
    paper_title: str,
    feature_values: List[str],
    expert_names: Optional[List[str]] = None
) -> str:
    """Build prompt for consolidating feature values across chunks and experts
    
    Args:
        feature_name: The name of the feature being consolidated
        paper_title: The title of the paper being analyzed
        feature_values: List of extracted values for this feature
        expert_names: Optional list of expert names who provided the values
    
    Returns:
        A prompt string for the LLM to consolidate the feature values
    """
    # Filter out empty or "Not Available" values
    valid_values = [value for value in feature_values if value and value != "Not Available"]
    
    # Format the values list with attribution to experts if available
    if expert_names and len(expert_names) == len(feature_values):
        values_list = "\n".join([f"- From {expert_names[i]}: {value}" 
                                for i, value in enumerate(valid_values)])
    else:
        values_list = "\n".join([f"- {value}" for value in valid_values])
    
    if not valid_values:
        return f"No valid values found for feature '{feature_name}' in paper '{paper_title}'. Respond with 'Not Available'."
    
    return f"""You are a scientific data consolidation expert working on a State of the Art (SOTA) table for research papers.

TASK:
Consolidate multiple extracted values for the feature '{feature_name}' from the paper '{paper_title}'.

EXTRACTED VALUES:
{values_list}

INSTRUCTIONS:
1. Synthesize these values into a single, coherent summary
2. Preserve all quantitative measurements, metrics, and technical details
3. When values conflict:
   - Prioritize the most specific and detailed information
   - If numerical values differ, use the most rigorous or recent methodology's result 
   - If descriptions conflict, explain the discrepancy briefly in your response
4. Keep the consolidated value concise (ideally 1-3 sentences) while being comprehensive
5. Maintain technical accuracy and use domain-specific terminology appropriately
6. If values mention specific implementations, methods, or metrics, include those details

EXAMPLES:
- For a feature named "Model Architecture", include the exact architecture type, key components, layers, and any novel adjustments
- For a feature named "Accuracy", include the precise percentage, the dataset used, and any qualifications
- For a feature named "Dataset Size", provide the exact numbers with appropriate units

Provide only the consolidated value as a string response without any prefacing text."""
