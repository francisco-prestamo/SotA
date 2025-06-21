def scraper_selection_prompt(research_description: str, scraper_infos: list, result_schema) -> str:
    """
    Build prompt for selecting appropriate scrapers based on research description.

    Args:
        research_description: Detailed description of what research topic/information is needed
        scraper_infos: List of available scrapers with their descriptions
        result_schema: Pydantic schema for the expected response format
    """
    scraper_descriptions = "\n".join([
        f"• {info['name']}: {info['description']}"
        for info in scraper_infos
    ])

    return f"""You are an expert research assistant tasked with selecting the most appropriate document scrapers for a specific research need.

RESEARCH DESCRIPTION:
{research_description}

AVAILABLE SCRAPERS:
{scraper_descriptions}

TASK:
Analyze the research description and determine which scrapers would be most effective for finding relevant academic content. For each scraper, evaluate:

1. **Relevance**: Does this scraper's content domain align with the research topic?
2. **Coverage**: Will this scraper likely contain the type of documents needed?
3. **Quality**: Does this scraper provide access to high-quality, peer-reviewed sources?
4. **Specificity**: Is this scraper specialized enough to find targeted research?

SELECTION CRITERIA:
- Choose scrapers that best match the research domain and methodology described
- Prioritize academic databases and repositories for scholarly research
- Consider both broad coverage and specialized sources when appropriate
- Avoid scrapers that are clearly irrelevant to the research focus

INSTRUCTIONS:
For each scraper, decide whether it should be used (true) or not (false), and provide a concise reasoning for your decision. Focus on how well each scraper aligns with the specific research needs described.

Return your response as a JSON object following this exact schema:
{result_schema.model_json_schema()}

Ensure your reasoning is specific to why each scraper would or would not be valuable for this particular research description."""