import json
from pydantic import BaseModel, Field
from typing import List

from board.board import ThesisKnowledgeModel


class ThesisAssessmentModel(BaseModel):
    is_sufficient: bool = Field(
        description="Whether the current knowledge about the thesis topic is sufficient"
    )
    reasoning: str = Field(description="Reasoning behind the assessment")
    missing_aspects: List[str] = Field(
        description="List of aspects that are missing in the current knowledge"
    )
    suggested_questions: List[str] = Field(
        description="Suggested questions to ask to gather more knowledge", default=[]
    )


def thesis_assessment_prompt(thesis_knowledge: ThesisKnowledgeModel, messages=None) -> str:
    keyknowledge_points = "".join(f"- {t}\n" for t in thesis_knowledge.thoughts)
    chat_history = ""
    if messages:
        chat_history = "\nChat history so far (user and receptionist):\n" + "\n".join(
            f"{m['sender'].capitalize()}: {m['content']}" for m in messages
        ) + "\n"
    prompt = (
        (
            f"""
You are an expert assistant, part of a system specialized in building state of the art sections for research papers,
the system should be able to investigate sources based on the description of a research paper and retrieve documents
that conform the state of the art in the different fields of research it covers. Your current task is much simpler 
than that: your objective is to assess the specific fields of research of the user's research paper, in order to 
discern which domain experts should be recruited so they can compile the state of the art for each field.

Currently, you have gathered the following information about the user's paper:
Paper Description: {thesis_knowledge.description}

Collected Knowledge Points:
{keyknowledge_points}
{chat_history}

CRITICAL STOPPING CONDITIONS - READ CAREFULLY:
1. FIRST, analyze the chat history for ANY indication that the user doesn't know more information. Look for phrases like:
   - "I don't know"
   - "I'm not sure"
   - "I'm still exploring"
   - "I haven't decided yet"
   - "I don't have more details"
   - Any expression of uncertainty or lack of knowledge

2. IF the chat history shows the user has indicated they don't know more details about their research, then IMMEDIATELY set is_sufficient=True and DO NOT ask any more questions, regardless of missing information.

3. The user saying they don't know IS A VALID ENDPOINT. Don't keep pushing for information they've already said they don't have.

4. Only ask questions if:
   - The user has NOT expressed uncertainty or lack of knowledge
   - AND the information is something the USER should reasonably know about their own research
   - AND it's NOT something the SYSTEM should determine automatically

5. NEVER ask questions about system responsibilities like expert recruitment, source identification, or benchmark selection.

Based on the above information, assess whether we have sufficient knowledge about the thesis topic to recommend expert reviewers and surveys. REMEMBER: If the user has already expressed they don't know more details, then the current knowledge IS sufficient to proceed.

Examples of Sufficient vs. Insufficient Knowledge:

Example Knowledge:
Description: A study on deep learning methods for medical computer vision tasks, with accurate testing of results
Knowledge points:
- This thesis is about deep learning for computer vision
- Testing through different benchmarks is planned
- The thesis is focused on medical applications

Assessment:"""
        )
        + ThesisAssessmentModel(
            is_sufficient=False,
            reasoning=(
                "While the general topic is known, critical research details are missing that only the user can provide. "
                "We need to know the specific medical domain and the user's planned methodology to recruit appropriate experts. "
                "However, we should not ask about specific benchmarks as the system can identify relevant ones automatically."
            ),
            missing_aspects=[
                "Specific medical domain or application area",
                "Planned deep learning methodologies or approaches",
                "Research objectives and expected outcomes"
            ],
            suggested_questions=[
                "What specific medical domain will your research focus on? (e.g., radiology, pathology, dermatology)",
                "What deep learning approaches are you planning to use or investigate?",
                "What are the main research objectives you want to achieve?"
            ],
        ).model_dump_json(indent=2)
        + """

Example with user indicating limited knowledge:
Chat History:
User: I'm working on machine learning for healthcare
Receptionist: What specific healthcare applications are you targeting?
User: I'm not sure yet, I'm still exploring the field

Assessment:"""
        + ThesisAssessmentModel(
            is_sufficient=True,
            reasoning=(
                "The user has explicitly stated 'I'm not sure yet' and 'I'm still exploring', which are clear indicators "
                "that they don't have more specific information to provide. The general domain (machine learning for healthcare) "
                "is sufficient to begin expert recruitment. Continuing to ask questions would be unproductive since the user "
                "has already expressed uncertainty. This is a valid stopping point."
            ),
            missing_aspects=[],
            suggested_questions=[],
        ).model_dump_json(indent=2)
        + """

Example with user saying they don't know:
Chat History:
User: I want to research artificial intelligence
Receptionist: What specific AI applications are you interested in?
User: I don't know exactly, maybe something with data analysis
Receptionist: What type of data analysis methods do you plan to use?
User: I don't know, I'm just starting

Assessment:"""
        + ThesisAssessmentModel(
            is_sufficient=True,
            reasoning=(
                "The user has clearly stated 'I don't know' multiple times and mentioned they are 'just starting'. "
                "This is a definitive signal to stop asking questions. The information provided (AI for data analysis) "
                "gives enough direction for initial expert recruitment. Further questioning is inappropriate when the "
                "user has explicitly indicated lack of knowledge."
            ),
            missing_aspects=[],
            suggested_questions=[],
        ).model_dump_json(indent=2)
        + """
Example Knowledge:
Description: An investigation into the use of composite materials for next-generation hypersonic aircraft structures, focusing on thermal resistance and structural integrity at extreme velocities

Knowledge points:
- The thesis is in the field of aerospace engineering
- It specifically focuses on hypersonic aircraft design
- It involves the use of composite materials
- Key performance concerns include thermal resistance and structural integrity
- The study is interested in extreme velocity flight conditions

Assessment:"""
        + ThesisAssessmentModel(
            reasoning="The research topic is clearly within aerospace engineering and identifies specific areas of focus: hypersonic aircraft, composite materials, and their performance under thermal and structural stress. These are well-defined subfields, and we can confidently recruit experts in hypersonic systems, aerospace materials, and high-temperature structural analysis.",
            is_sufficient=True,
            missing_aspects=[],
            suggested_questions=[],
        ).model_dump_json(indent=2)
        + f"""

Now, please assess the current thesis knowledge and determine if it's sufficient for the purpose of recruiting domain experts who can compile the state of the art for the relevant research fields.

MANDATORY CHECKLIST BEFORE RESPONDING:
□ Have I checked the chat history for expressions of uncertainty or "I don't know"?
□ If the user expressed uncertainty, am I setting is_sufficient=True and asking NO questions?
□ Am I only asking questions if the user has NOT expressed lack of knowledge?
□ Are my questions about USER knowledge, not SYSTEM responsibilities?

Remember: A user saying "I don't know" is a STOP signal, not a request for more questions.

Output your assessment following this schema:
{json.dumps(ThesisAssessmentModel.model_json_schema(), indent=2)}
"""
    )

    return prompt