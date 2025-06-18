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


def thesis_assessment_prompt(thesis_knowledge: ThesisKnowledgeModel) -> str:
    prompt = (
        (
            f"""
You are an expert assistant, part of a system specialized in building state of the art sections for research papers,
the system should be able to investigate sources based on the description of a research paper and retrieve documents
that confom the state of the art in the different fields of research it covers. Your current task is much simpler 
than that: your objective is to assess the specific fields of research of the user's research paper, in order to 
discern which domain experts should be recruited so they can compile the state of the art for each field.

Currently, you have gathered the following information about the user's paper:
Paper Description: {thesis_knowledge.description}

Collected Knowledge Points:
{"".join(f"- {t}\n" for t in thesis_knowledge.thoughts)}

Based on the above information, assess whether we have sufficient knowledge about the thesis topic to recommend expert reviewers and surveys, if the information isn't enough, you must output a set of missing aspects in the knowledge of the
paper, and based on those a set of questions to ask the user to better clarify their intentions

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
                "While the general topic is known, specific research questions and methodologies are not specified, "
                "for example, while we know the applications are to be medical in nature, we do not know any specific "
                "field of medicine the thesis will focus on, similarly, we do not know any specific benchmarks to be "
                "used for evaluation, and we would need clarification in the previous aspect to be able to decide this"
            ),
            missing_aspects=[
                "Field of medicine of focus (e.g. cancer research, psychological assistance)",
                "What specific benchmarks are to be used to assess the quality of the solutions",
                "Technologies and methods to be used for the computer vision tasks",
            ],
            suggested_questions=[
                "What specific field of medicine will your research focus on? (e.g. cancer research, psychological assistance)",
                "What specific benchmarks are you planning to employ?",
                "What methods are you using for the computer vision tasks?",
                "What medical problems do you plan on solving?",
            ],
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
Now, please assess the current thesis knowledge and determine if it's sufficient for the purpose of assessing which experts would cover its themes and topics
If it's not sufficient, please suggest questions to ask the user to gather more knowledge.

Output your assessment following this schema:
{json.dumps(ThesisAssessmentModel.model_json_schema(), indent=2)}
"""
    )

    return prompt
