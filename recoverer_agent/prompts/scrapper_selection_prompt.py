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

    return f"""You are a research query specialist. Your task is to select the most relevant document sources and craft precise search queries that will retrieve the most targeted academic content.

RESEARCH OBJECTIVE:
{research_description}

AVAILABLE SOURCES:
{scraper_descriptions}

YOUR MISSION:
1. Select sources that directly align with the research domain
2. Generate laser-focused search queries that capture the core research intent

SELECTION STRATEGY:
• Prioritize sources with strong domain alignment over general coverage
• Choose specialized databases for niche research areas
• Include 2-4 complementary sources for comprehensive coverage
• Skip sources with minimal relevance to avoid noise

QUERY CRAFTING PRINCIPLES:
• Use 3-5 strategic keywords that encapsulate the research essence
• Combine domain-specific terms with methodological keywords
• Target the language researchers actually use in titles and abstracts
• Avoid generic terms that produce too many irrelevant results
• No operators, quotes, or complex syntax - keep it clean and searchable

QUERY EXAMPLES BY RESEARCH TYPE:
• Technical research: "convolutional neural networks image classification", "variational quantum algorithms optimization", "transformer architecture attention mechanisms"
• Social sciences: "participatory urban design community engagement", "loss aversion behavioral finance", "social network analysis political polarization"
• Medical research: "CAR-T cell therapy leukemia", "CRISPR Cas9 therapeutic applications", "mRNA vaccine immunogenicity"
• Environmental: "lithium-ion battery energy storage", "carbonic acid coral calcification", "photovoltaic efficiency perovskite materials"
• Business/Economics: "supply chain resilience risk management", "digital transformation organizational change", "cryptocurrency market volatility"
• Psychology/Neuroscience: "working memory prefrontal cortex", "cognitive behavioral therapy depression", "neuroplasticity rehabilitation therapy"

OPTIMIZATION FOCUS:
Each query should be specific enough to filter out irrelevant papers while broad enough to capture variations in terminology. Think about what keywords would appear in the title or abstract of your ideal research paper.

For each selected source, provide:
- Clear rationale for why it's relevant to the research
- A targeted search query optimized for that source's content domain
- Confidence level in the query's effectiveness (if schema includes this)

Remember: The goal is precision over quantity. Better to have fewer, highly relevant results than many irrelevant ones."""
