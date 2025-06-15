def thesis_assessment_prompt(thesis_knowledge, assessment_model):
    """
    Prompt for determining if the current knowledge about a thesis topic is sufficient.
    
    Args:
        thesis_knowledge: The current knowledge about the thesis topic
        assessment_model: The model class for the assessment
        
    Returns:
        str: The formatted prompt
    """
    examples = [
        {
            "knowledge": {
                "thoughts": ["This thesis is about deep learning for computer vision",
                             "Several papers mention transformers for image recognition",
                             "There are benchmark datasets like ImageNet"],
                "description": "A study on deep learning methods for computer vision tasks"
            },
            "assessment": {
                "is_sufficient": False,
                "reasoning": "While the general topic is known, specific research questions, methodologies, and recent advancements are not clear.",
                "missing_aspects": ["Specific research questions", 
                                    "Current state-of-the-art methods", 
                                    "Application domains",
                                    "Evaluation metrics"]
            }
        },
        {
            "knowledge": {
                "thoughts": ["This thesis focuses on natural language processing",
                             "The main focus is on large language models",
                             "Specific techniques like transfer learning, fine-tuning are mentioned",
                             "Several benchmark datasets are discussed: GLUE, SuperGLUE",
                             "Research questions include: How can LLMs be made more efficient?",
                             "Methodologies like distillation and pruning are mentioned",
                             "Evaluation will be based on BLEU, ROUGE, and human evaluation"],
                "description": "A comprehensive study on improving efficiency of large language models"
            },
            "assessment": {
                "is_sufficient": True,
                "reasoning": "The knowledge covers the topic comprehensively including research questions, methodologies, evaluation metrics, and current state-of-the-art.",
                "missing_aspects": []
            }
        }
    ]
    
    prompt = f"""As an AI thesis advisor, your task is to determine if we have sufficient knowledge about a thesis topic to recommend expert reviewers and surveys.

Here is the current knowledge about the thesis:

Thesis Description: {thesis_knowledge.description}

Collected Knowledge Points:
"""
    
    for thought in thesis_knowledge.thoughts:
        prompt += f"- {thought}\n"
    
    prompt += """
Based on the above information, assess whether we have sufficient knowledge about the thesis topic to recommend expert reviewers and surveys.

Examples of Sufficient vs. Insufficient Knowledge:

"""

    for example in examples:
        prompt += f"Example Knowledge:\n"
        prompt += f"Description: {example['knowledge']['description']}\n"
        prompt += "Knowledge points:\n"
        for thought in example['knowledge']['thoughts']:
            prompt += f"- {thought}\n"
        prompt += f"\nAssessment:\n"
        prompt += f"Is Sufficient: {example['assessment']['is_sufficient']}\n"
        prompt += f"Reasoning: {example['assessment']['reasoning']}\n"
        prompt += "Missing Aspects:\n"
        for aspect in example['assessment']['missing_aspects']:
            prompt += f"- {aspect}\n"
        prompt += "\n---\n\n"

    prompt += """
Now, please assess the current thesis knowledge and determine if it's sufficient.
If it's not sufficient, please suggest questions to ask the user to gather more knowledge.

Output your assessment following the provided model schema.
"""
    
    return prompt
