import json
from typing import Type
from pydantic import BaseModel, Field


class DescriptionUpdate(BaseModel):
    """
    Models the update to the research paper description based on user answers
    """

    reasoning: str = Field(
        description="The reasoning behind the changes made to the description, explaining how the user's answers were incorporated"
    )
    updated_description: str = Field(
        description="The updated description of the research paper, incorporating the new information from the user's answers"
    )


def update_description_prompt(
    current_description: str,
    questions_asked: str,
    user_answers: str,
) -> str:
    """
    Creates a prompt for updating the research paper description based on user answers.

    Args:
        current_description: The current description of the research paper
        questions_asked: The questions that were asked to the user
        user_answers: The answers provided by the user

    Returns:
        A prompt string that guides the LLM to update the description
    """
    return f"""
You are tasked with updating the description of a research paper based on new information provided by the user.
Your goal is to incorporate the user's answers into the description while maintaining its coherence and completeness.

Current description of the research paper:
{current_description}

Questions that were asked to the user:
{questions_asked}

Answers provided by the user:
{user_answers}

Please update the description by:
1. Incorporating the new information from the user's answers
2. Maintaining the overall structure and flow of the description
3. Ensuring all important details from the original description are preserved
4. Making the description more precise and complete based on the new information

The output should contain:
1. The updated description that incorporates the new information
2. A clear explanation of how the user's answers were used to update the description

Output only the json object, no additional text.
"""

