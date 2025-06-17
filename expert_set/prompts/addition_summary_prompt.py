from typing import List

def build_addition_summary_prompt(
    added_titles: List[str],
    expert_names: List[str],
    new_features: List[str]
) -> str:
    """Build prompt for summarizing the paper addition process"""
    titles_list = "\n".join([f"- {title}" for title in added_titles])
    experts_list = ", ".join(expert_names)
    features_list = "\n".join([f"- {feature}" for feature in new_features]) if new_features else "None"
    
    return f"""Summarize the impact of adding new papers to the SOTA table.

PAPERS ADDED:
{titles_list}

EXPERTS INVOLVED:
{experts_list}

NEW FEATURES IDENTIFIED AND ADDED:
{features_list}

INSTRUCTIONS:
1. Summarize the contribution of the newly added papers
2. Highlight the value of any new features that were identified
3. Explain how this enhances the SOTA table and thesis knowledge
4. Keep the summary concise but informative (2-3 sentences)

Provide a summary as a string response."""
