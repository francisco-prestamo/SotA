def build_expert_search_reasoning_prompt(sota_md: str, thesis_desc: str, thesis_thoughts: list[str], expert_context: dict) -> str:
    return (
        f"# SOTA Table\n{sota_md}\n\n"
        f"# Thesis Knowledge\nDescription: {thesis_desc}\nThoughts: {thesis_thoughts}\n\n"
        f"For each expert, analyze what is missing from the SOTA table to satisfy the thesis knowledge.\n"
        f"For each, provide reasoning and what to search for.\n\n"
        f"Experts: {expert_context}\n"
    )
