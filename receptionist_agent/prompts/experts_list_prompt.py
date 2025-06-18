from board.board import ThesisKnowledgeModel
from ..models import BuildExpertCommandList


def experts_list_prompt(thesis_knowledge: ThesisKnowledgeModel):
    """
    Prompt for generating a list of experts based on thesis knowledge.
    
    Args:
        thesis_knowledge: The current knowledge about the thesis topic
        experts_list_model: The model class for the experts list
        
    Returns:
        str: The formatted prompt
    """
    prompt = f"""As an AI thesis advisor, your task is to generate a list of expert descriptions based on the thesis topic.
These experts will be consulted for State-of-the-Art surveys in their respective fields related to the thesis.

Here is the knowledge about the thesis:

Thesis Description: {thesis_knowledge.description}

Collected Knowledge Points:
"""
    
    for thought in thesis_knowledge.thoughts:
        prompt += f"- {thought}\n"
    
    prompt += f"""
Based on the above information, generate a list of 3-5 experts who would be ideal to provide surveys about the state-of-the-art in the thesis topic.

For each expert, provide:
1. A name or identifier (can be a fictional name representing an expert in a specific subfield)
2. A brief description of their expertise and background
3. A specific query that would be used to search for surveys they might have authored or would be knowledgeable about

Expected Output:
```
{BuildExpertCommandList.model_json_schema()}
```

Generate a diverse list of experts covering different aspects of the thesis topic. The experts should be complementary, not redundant.
Output your list following the provided model schema.
"""
    
    return prompt
