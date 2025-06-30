def scrapper_selection_prompt(research_description: str, scraper_infos: list, result_schema) -> str:
    """
    Build prompt for selecting the best scrapers and generating optimal search queries based on research description.
    Args:
        research_description: Detailed description of what research topic/information is needed
        scraper_infos: List of available scrapers with their descriptions and previous searches
        result_schema: Pydantic schema for the expected response format
    """
    # Generate schema information for the model
    schema_info = """
    The JSON response should follow this structure:
    {
        "reasoning": "Overall reasoning for your selections",
        "selections": [
            {
                "source_name": "Name of the scraper",
                "selected": true/false,
                "queries": [
                    {
                        "query": "specific search query 1",
                        "reasoning": "why this query is valuable"
                    },
                    {
                        "query": "specific search query 2",
                        "reasoning": "why this query is valuable"
                    },
                    ...up to 3 queries per source
                ],
                "source_reasoning": "Why this source is relevant or not relevant"
            },
            ...one object per available source
        ]
    }
    """
    
    scraper_descriptions = "\n".join([
        f"• {info['name']}: {info['description']}\n  Previous searches: {', '.join(info.get('previous_searches', ['None']))}"
        for info in scraper_infos
    ])
    
    return f"""You are a research query specialist. Your task is to identify the most relevant sources and craft precise search queries for finding academic papers directly relevant to the research description.

RESEARCH OBJECTIVE:
{research_description}

AVAILABLE SOURCES WITH PREVIOUS SEARCHES:
{scraper_descriptions}

YOUR FLEXIBLE MISSION:
1. Determine the MOST RELEVANT sources for the current research objective (1-3 sources)
2. For each selected source, generate UP TO 3 OPTIMAL search queries tailored to its strengths
3. Ensure all queries directly address different aspects of the research description
4. Take into account previous searches to avoid duplication

MULTI-QUERY STRATEGY:
• Identify multiple aspects or angles of the research objective that should be explored
• For each selected source, create up to 3 complementary queries that address different aspects
• Ensure each query focuses on distinct methodologies, applications, or theoretical frameworks
• Make each query specialized and non-overlapping with others to maximize coverage
• Balance between depth (specific techniques) and breadth (different approaches) in your query set

QUERY CRAFTING PRINCIPLES FOR SCIENTIFIC DATABASES:
• Use precise technical terminology that academic researchers would use in papers
• Include specific methodologies, algorithms, metrics, or frameworks in each query
• Target recent advances and state-of-the-art approaches where appropriate
• Use 3-6 highly specific technical terms in each query rather than general descriptions
• Match terminology to the indexing patterns of each selected scientific database

TAILORING QUERIES TO SOURCE STRENGTHS:
• For arXiv: Focus on recent technical papers with specific algorithm/method names
• For PubMed: Use precise medical/biological terminology and include methodology terms
• For academic databases: Include specific metrics, evaluation approaches, or frameworks
• For Semantic Scholar: Balance technical precision with broader conceptual terms

EXAMPLES OF MULTI-QUERY STRATEGY:

Example 1: Research description about "deep learning for medical image analysis"
Source: arXiv
Query 1: "convolutional neural networks segmentation medical imaging"
Query 2: "self-supervised learning anatomical structure detection"
Query 3: "uncertainty estimation diagnostic accuracy medical images"
Reasoning: These queries cover different technical approaches (CNNs, self-supervised learning) and challenges (segmentation, detection, uncertainty estimation) in medical imaging.

Example 2: Research description about "recommendation systems addressing cold start problem"
Source: Semantic Scholar
Query 1: "hybrid recommendation systems cold start problem"
Query 2: "knowledge graph embeddings user preferences new items"
Query 3: "few-shot learning recommendation systems evaluation metrics"
Reasoning: These queries address the main challenge (cold start) with different approaches (hybrid systems, knowledge graphs, few-shot learning).

Example 3: Research description about "reinforcement learning for robotic manipulation"
Source 1: arXiv
Query 1: "reinforcement learning sample efficiency robotic grasping"
Query 2: "sim2real transfer robotic manipulation dexterous hands"
Source 2: Semantic Scholar
Query 1: "hierarchical reinforcement learning complex manipulation tasks"
Reasoning: These queries cover both technical challenges (sample efficiency, sim2real) across multiple relevant sources.

DETECTING MULTIPLE IMPORTANT ASPECTS TO SEARCH:
• Identify several technically significant components in the research objective
• For each component, focus on specialized methodologies that address different needs or challenges
• Consider various performance bottlenecks that would be addressed in recent literature
• Target different evaluation methodologies or metrics for assessing solutions
• Ensure your queries together provide comprehensive coverage of the research topic

BUILDING ON PREVIOUS SEARCHES:
• Analyze previous queries to avoid duplication and ensure progress
• If previous searches used general terms, create more technically specific queries
• If previous searches focused on methods, consider targeting applications or evaluations
• Formulate queries that address limitations or challenges identified in previously found papers
• Create a complementary set of queries that explore different angles of the research topic

OUTPUT REASONING:
For each source and its associated queries, provide:
1. Why this source is relevant for the current research objective
2. Why each specific query is valuable at this moment in the research process
3. How each query complements the others to provide comprehensive coverage
4. How the crafted queries use specific technical terminology suited to that source's indexing
5. How these queries relate to or improve upon previous searches on this source

SEARCH QUERY FORMAT REQUIREMENTS:
• Use only the most precise technical terminology relevant to the current research need
• Keep each query concise (3-6 key technical terms)
• No quotes, Boolean operators or special characters
• Focus on technical precision rather than natural language phrasing
• Prioritize terms that would appear in academic paper titles and abstracts
• Ensure each query for a source explores a different aspect of the research topic

OUTPUT FORMAT:
{schema_info}"""