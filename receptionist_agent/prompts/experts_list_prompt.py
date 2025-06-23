import json
from board.board import ThesisKnowledgeModel
from ..models import BuildExpertCommandList


def experts_list_prompt(thesis_knowledge: ThesisKnowledgeModel) -> str:
    """
    Generate a prompt for creating a list of experts based on thesis knowledge.

    This function creates a detailed prompt that guides an AI to generate expert
    descriptions for conducting State-of-the-Art surveys related to a thesis topic.

    Args:
        thesis_knowledge (ThesisKnowledgeModel): The current knowledge about the thesis topic,
                                               including description and collected thoughts

    Returns:
        str: A formatted prompt string for expert list generation
    """

    # Build the knowledge points section
    knowledge_points = ""
    for thought in thesis_knowledge.thoughts:
        knowledge_points += f"- {thought}\n"

    prompt = f"""As an AI thesis advisor, your task is to generate a list of expert descriptions based on the thesis topic. These experts will be consulted for State-of-the-Art surveys in their respective fields related to the thesis.

Here is the knowledge about the thesis:

Thesis Description: {thesis_knowledge.description}

Collected Knowledge Points:
{knowledge_points}

Based on the above information, generate a list of 3-5 experts who would be ideal to provide surveys about the state-of-the-art in the thesis topic.

For each expert, provide:
1. A name or identifier (can be a fictional name representing an expert in a specific subfield)
2. A brief description of their expertise and background
3. A specific query that would be used to search for surveys they might have authored or would be knowledgeable about

IMPORTANT: Keep search queries simple and straightforward. Do NOT use operators like "AND", "OR", or quotation marks. Use natural language phrases that would work well in academic search engines.

Examples of good search queries:
- "machine learning healthcare applications survey"
- "deep learning medical imaging survey"
- "natural language processing clinical notes survey"
- "computer vision diagnostic systems survey"
- "reinforcement learning drug discovery survey"

Expected Output Format:
```json
{json.dumps(BuildExpertCommandList.model_json_schema(), indent=2)}
```

Generate a diverse list of experts covering different aspects of the thesis topic. The experts should be complementary, not redundant. Each expert should represent a distinct area of expertise that contributes to understanding the broader thesis landscape.

Guidelines:
- Focus on experts who would have published survey papers or comprehensive reviews
- Ensure each expert covers a unique aspect of the thesis topic
- Make search queries specific enough to find relevant surveys but broad enough to capture comprehensive coverage
- Use terminology that would appear in academic paper titles and abstracts
"""

    return prompt