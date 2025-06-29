def scrapper_selection_prompt(research_description: str, scraper_infos: list, result_schema) -> str:
    """
    Build prompt for selecting appropriate scrapers and generating search queries based on research description.
    Args:
        research_description: Detailed description of what research topic/information is needed
        scraper_infos: List of available scrapers with their descriptions and previous searches
        result_schema: Pydantic schema for the expected response format
    """
    scraper_descriptions = "\n".join([
        f"• {info['name']}: {info['description']}\n  Previous searches: {', '.join(info.get('previous_searches', ['None']))}"
        for info in scraper_infos
    ])
    
    return f"""You are a research query specialist. Your task is to select ALL relevant document sources and craft comprehensive search queries that will retrieve the most complete academic content coverage on the subject.

RESEARCH OBJECTIVE:
{research_description}

AVAILABLE SOURCES WITH PREVIOUS SEARCHES:
{scraper_descriptions}

YOUR ENHANCED MISSION:
1. Select ALL sources that could contribute to a complete understanding of the research topic
2. Generate comprehensive search queries that cover the entire scope of the research intent
3. Consider complementary angles and related subtopics to ensure complete coverage
4. Take into account previous searches to avoid duplication and expand knowledge

SELECTION STRATEGY (COMPREHENSIVE APPROACH):
• Select ALL sources with ANY relevance to the research domain
• Utilize BOTH specialized databases AND general sources for maximum coverage
• Ensure NO relevant information source is excluded
• When in doubt about a source's relevance, INCLUDE it rather than exclude

COMPREHENSIVE QUERY CRAFTING PRINCIPLES:
• Use natural language search terms without operators (no quotes, AND, OR, NOT, etc.)
• Craft BRIEF queries using 4-8 terms per query
• Keep queries concise and focused - avoid long phrases or sentences
• Combine core research terms with adjacent concepts in brief plain text format
• Ensure queries cover an specific research area about the research topic
• Create targeted queries that will find papers published from different perspectives
• Use straightforward, concise language that search engines can interpret naturally
• Consider how the topic might be discussed in different academic disciplines
• Avoid complex search operators and lengthy descriptions - rely on brief natural keyword matching

BUILDING ON PREVIOUS SEARCHES:
• Review previous search queries for each source and build upon them
• If previous searches were narrow, add broader perspective queries
• If previous searches were broad, add more specific targeted queries
• Create queries that explore different aspects not covered in previous searches
• Consider how to expand or refine previous searches to get complementary results

QUERY EXAMPLES FOR COMPREHENSIVE COVERAGE (BRIEF, NO OPERATORS):
• Technical research: "artificial intelligence machine learning neural networks", "AI algorithms deep learning applications", "machine learning survey recent advances", "AI challenges ethical considerations"
• Social sciences: "social media mental health adolescents", "social media cultural differences behavior", "social media longitudinal studies outcomes", "social media policy regulation effects"
• Medical research: "diabetes treatment insulin therapy", "diabetes systematic review meta analysis", "diabetes clinical trials patient outcomes", "diabetes prevention lifestyle interventions"
• Interdisciplinary topics: Use 4-8 terms from relevant fields as separate focused queries

SEARCH QUERY FORMAT REQUIREMENTS:
• Use only natural language keywords and phrases
• Keep queries BRIEF - 4-8 key terms per query
• No quotation marks, Boolean operators, or special characters
• Separate concepts with spaces, not operators
• Keep queries focused and conversational
• Avoid lengthy phrases or complete sentences
• Balance specificity with breadth in term selection
• Example: "climate change ocean acidification marine ecosystems" instead of overly long descriptions

COMPREHENSIVE COVERAGE STRATEGY:
Each query should aim to capture a different slice of the research landscape using brief, natural language search terms. Together, your queries should leave no stone unturned, ensuring all relevant papers are discovered regardless of the exact terminology they use.

For each selected source, provide:
- Clear explanation of how this source contributes to comprehensive understanding
- A brief natural language search query (4-8 terms, no operators) that captures aspects of the topic this source is best positioned to provide
- How this query complements or extends previous searches on this source

PRIORITY: The goal is COMPLETE COVERAGE using brief, simple search terms. It's better to retrieve some extra papers than to miss important relevant ones. When in doubt, be more inclusive in your source selection and query formulation."""