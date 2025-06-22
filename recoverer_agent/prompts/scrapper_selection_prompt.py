def scrapper_selection_prompt(research_description: str, scraper_infos: list, result_schema) -> str:
    """
    Build prompt for selecting appropriate scrapers and generating search queries based on research description.

    Args:
        research_description: Detailed description of what research topic/information is needed
        scraper_infos: List of available scrapers with their descriptions
        result_schema: Pydantic schema for the expected response format
    """
    scraper_descriptions = "\n".join([
        f"• {info['name']}: {info['description']}"
        for info in scraper_infos
    ])

    return f"""You are an expert research assistant tasked with selecting the most appropriate document scrapers for a specific research need and generating effective search queries for each selected scraper.

RESEARCH DESCRIPTION:
{research_description}

AVAILABLE SCRAPERS:
{scraper_descriptions}

TASK:
Analyze the research description and determine which scrapers would be most effective for finding relevant academic content. For each scraper, you need to:

1. **Decide if it should be used**: Evaluate relevance, coverage, quality, and specificity
2. **Generate a search query**: If selected, create a concise, targeted search query

SELECTION CRITERIA:
- Choose scrapers that best match the research domain and methodology described
- Prioritize academic databases and repositories for scholarly research
- Consider both broad coverage and specialized sources when appropriate
- Avoid scrapers that are clearly irrelevant to the research focus

QUERY GENERATION GUIDELINES:
- Create concise, specific search queries (2-6 words typically)
- Focus on key terms and concepts from the research description
- Avoid operators like "OR", "AND", quotation marks, or complex syntax
- Use simple keyword combinations that capture the essence of the research
- Tailor each query to the specific scraper's content domain

INSTRUCTIONS:
For each scraper, decide whether it should be used (true/false) and if true, provide a targeted search query. Include reasoning for your decisions focusing on how well each scraper aligns with the research needs.

Return your response as a JSON object following this exact schema:
{result_schema.model_json_schema()}

Example query formats:
- "machine learning healthcare"
- "climate change adaptation"
- "neural networks optimization"
- "sustainable energy systems"

Ensure queries are simple, relevant, and optimized for finding the most pertinent research documents."""