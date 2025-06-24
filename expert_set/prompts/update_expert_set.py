import json
from typing import Type, List, Optional
from pydantic import BaseModel, Field


class ExpertSetUpdate(BaseModel):
    """
    Models the update to the expert set based on research paper description and user interactions
    """

    whether_to_remove_reasoning: str = Field(
        description="The reasoning behind removing experts from the set, explaining why specific experts are no longer suitable.",
        examples=[
            "The removed experts primarily focus on outdated statistical models, which do not align with the deep learning techniques emphasized in the paper.",
            "No experts are redundant or unnecessary to understand the research paper, so none should be removed"
        ],
    )
    to_remove: Optional[List[str]] = Field(
        default=None,
        description="List of expert IDs to remove from the set, or None if no experts should be removed",
        examples=[["expert_2", "expert_1"]],
    )
    whether_to_add_reasoning: str = Field(
        description="The reasoning behind why experts should or should not be added to the set, explaining why additional expertise is needed, or why not.",
        examples=[
            "The current expert set lacks coverage in distributed systems, which is now a key focus of the paper. Adding a distributed systems expert to evaluate scalability aspects.",
            "The current expert set has a broad enough coverage of the themes and topic treated in the research paper in question"
        ],
    )
    to_add: Optional[List[str]] = Field(
        default=None,
        description="List of new expert descriptions to add to the set, or None if no experts should be added.",
        examples=[
            [
                "Expert in transformer-based language models, specializing in attention mechanisms and model architecture optimization. Evaluates technical soundness and innovation in model design.",
                "Computer vision specialist with focus on object detection and segmentation. Expert in evaluating accuracy metrics and implementation efficiency.",
                "Natural language processing researcher with expertise in multilingual models and cross-lingual transfer learning. Assesses language coverage and transfer capabilities.",
                "Machine learning engineer specializing in distributed training systems and model optimization. Evaluates scalability and computational efficiency.",
            ]
        ],
    )


def update_expert_set_prompt(
    current_description: str,
    old_description: str,
    current_experts: List[str],
    questions_asked: str,
    user_answers: str,
) -> str:
    """
    Creates a prompt for updating the expert set based on research paper description and user interactions.

    Args:
        current_description: The current description of the research paper
        old_description: The previous description of the research paper
        current_experts: List of current expert descriptions
        questions_asked: The questions that were asked to the user
        user_answers: The answers provided by the user

    Returns:
        A prompt string that guides the LLM to update the expert set
    """
    return f"""
You are tasked with evaluating and potentially updating the set of experts for a research paper based on its description and user interactions.
Your goal is to ensure that all research fields mentioned in the description are adequately covered by the expert set.

Previous description of the research paper:
{old_description}

Current description of the research paper:
{current_description}

Current set of experts:
{json.dumps(current_experts, indent=2)}

Questions that were asked to the user:
{questions_asked}

Answers provided by the user:
{user_answers}

Please carefully evaluate whether the current expert set adequately covers all research fields mentioned in the description.
Consider the following guidelines:
1. Only add new experts if you are CERTAIN that there are research fields in the description that are not covered by the current experts
2. Only remove experts if you are CERTAIN that their expertise is no longer relevant or is redundant
3. New experts should not be redundant with existing experts or with each other
4. Each expert should have a clear, specific area of expertise that is relevant to the research paper
5. The expert set should be as small as possible while still covering all necessary research fields
6. Compare the old and new descriptions to identify any shifts in research focus that might require expert set changes

When adding new experts, follow these guidelines for expert descriptions:
- Be specific about the research field and subfield
- Include relevant methodologies or techniques they specialize in
- Mention their role in evaluating the research
- Keep descriptions concise but informative

The output should contain:
1. A clear explanation of why experts were added or removed (or why no changes were needed)
2. A list of expert IDs to remove (if any)
3. A list of new expert descriptions to add (if any), following the format of the examples above

Output only a json object, no additional text.
"""

