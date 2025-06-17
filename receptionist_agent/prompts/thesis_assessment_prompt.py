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
As an AI thesis advisor, your task is to determine if we have sufficient knowledge about a thesis topic to recommend expert reviewers and surveys.

Here is the current knowledge about the thesis:

Thesis Description: {thesis_knowledge.description}

Collected Knowledge Points:
{"".join(f"- {t}\n" for t in thesis_knowledge.thoughts)}

Based on the above information, assess whether we have sufficient knowledge about the thesis topic to recommend expert reviewers and surveys, if the information isn't enough, you must output a set of missing aspects in the knowledge of the
paper, and based on those a set of questions the user can ask you the answers of which would clarify the topic of their 
paper.

Examples of Sufficient vs. Insufficient Knowledge:

Example Knowledge:
Description: A study on deep learning methods for computer vision tasks
Knowledge points:
- This thesis is about deep learning for computer vision
- Several papers mention transformers for image recognition
- There are benchmark datasets like ImageNet

Assessment:"""
        )
        + ThesisAssessmentModel(
            is_sufficient=False,
            reasoning=(
                "While the general topic is known, specific research questions, "
                "methodologies, and recent advancements are not clear."
            ),
            missing_aspects=[
                "Specific research questions",
                "Current state‑of‑the‑art methods",
                "Application domains",
                "Evaluation metrics",
            ],
            suggested_questions=[
                "What are the current state‑of‑the‑art deep learning models used in vision tasks?",
                "What novel research questions are being posed in recent papers?",
                "Which application domains (e.g., medical imaging, autonomous vehicles) are most impacted?",
                "What evaluation metrics best reflect real‑world performance in these tasks?",
            ],
        ).model_dump_json(indent=2)
        + """
Example Knowledge:
Description: A comprehensive study on optimizing request latency and resource utilization in large‑scale, geo‑distributed microservice architectures. 
This work builds a novel latency‑aware sharding layer that integrates with Kubernetes to dynamically reassign microservice partitions based on 
real‑time network performance and workload distribution. A deep reinforcement‑learning agent continuously learns optimal shard placements, 
balancing the trade‑off between migration overhead and tail‑latency reduction. The system is evaluated under multi‑region deployment traces, 
measuring improvements in 99th‑percentile latency, cross‑region bandwidth consumption, and cost of VM migrations. Results aim to demonstrate 
significant latency gains over static and heuristic‑based approaches while maintaining cluster stability.

Knowledge points:
- This thesis tackles dynamic load balancing in geo‑distributed microservice clusters
- Existing solutions assume uniform network latency, but real‑world links vary by up to 300 ms
- Workload skews (e.g. user hotspots) cause hot partitions that current sharding schemes can’t mitigate
- Proposed system introduces a latency‑aware sharding layer atop Kubernetes’ scheduler
- Sharding decisions use a reinforcement‑learning agent trained on synthetic and trace‑driven workloads
- Controller adapts at 5 second intervals, trading off migration cost vs. request latency reduction
- Evaluation metrics include 99th‑percentile tail latency, cross‑region bandwidth usage, and VM‐spinup overhead
- Research questions: Can RL‑guided sharding outperform static heuristics? How does adaptation frequency impact stability?

Assessment:"""
        + ThesisAssessmentModel(
            reasoning=(
                "The proposal presents a clear problem statement, detailed pipeline (latency‑aware sharding, "
                "RL agent, adaptation cadence), and concrete evaluation metrics (tail latency, bandwidth usage, "
                "migration cost). It covers both algorithmic design and real‑world deployment concerns."
            ),
            is_sufficient=True,
            missing_aspects=[],
            suggested_questions=[],
        ).model_dump_json(indent=2)
        + f"""
Now, please assess the current thesis knowledge and determine if it's sufficient.
If it's not sufficient, please suggest questions to ask the user to gather more knowledge.

Output your assessment following this schema:
{json.dumps(ThesisAssessmentModel.model_json_schema(), indent=2)}
"""
    )

    return prompt
