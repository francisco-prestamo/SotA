def build_expert_search_reasoning_prompt(sota_md: str, thesis_desc: str, thesis_thoughts: list[str], expert_context: dict) -> str:
    # Format thesis thoughts as a bulleted list if there are any
    formatted_thoughts = ""
    if thesis_thoughts:
        formatted_thoughts = "Thoughts:\n" + "\n".join([f"- {thought}" for thought in thesis_thoughts])
    else:
        formatted_thoughts = "No additional thesis thoughts provided."
    
    return (
        f"# Current State-of-the-Art (SOTA) Table\n{sota_md}\n\n"
        f"# Thesis Knowledge\nDescription: {thesis_desc}\n{formatted_thoughts}\n\n"
        f"# Expert Analysis Task\n"
        f"Analyze the gaps between the current SOTA table and the thesis requirements. For each expert in the provided context, "
        f"determine what critical information is missing from the current SOTA table that would strengthen the thesis analysis.\n\n"
        f"## Instructions:\n"
        f"1. Carefully evaluate the areas where the SOTA table lacks sufficient depth, breadth, or specific technical details needed for the thesis.\n"
        f"2. Identify the most important missing information categories (methodologies, performance metrics, dataset comparisons, etc.).\n"
        f"3. For each expert, provide specific reasoning on how their domain knowledge can address these gaps.\n"
        f"4. Suggest concrete search queries that will help retrieve the missing information.\n"
        f"5. Prioritize searches that would yield the most impactful additions to the SOTA table.\n\n"
        f"## Response Format (for each expert):\n"
        f"- **Gap Analysis**: What critical information is missing from the current SOTA table?\n"
        f"- **Reasoning**: Why is this information important for the thesis?\n"
        f"- **Search Recommendation**: Specific search terms or phrases to retrieve this information.\n"
        f"- **Expected Impact**: How will filling this gap strengthen the thesis analysis?\n\n"
        f"## Available Experts:\n{expert_context}\n"
    )
