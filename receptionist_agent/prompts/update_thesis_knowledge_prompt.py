from typing import List
from pydantic import BaseModel
from board.board import ThesisKnowledgeModel

class QAndA(BaseModel):
    system_question: str
    user_answer: str

def update_thesis_knowledge_prompt(current_knowledge: ThesisKnowledgeModel, qa_pairs: List[QAndA]):
    prompt = f"""
You are an expert system, tasked with interacting with a user with the objective of discerning the subject
and themes of their research paper, as part of this process, you proceeded to ask the user a set of 
questions in order to better better understand their purposes.

Currently, this is the knowledge you have about the research paper
### Knowledge ###
Description: {current_knowledge.description}

"""
    for thought in current_knowledge.thoughts:
        prompt += f"- {thought}\n"
    prompt += f"""
### ###

This is the process of questions asked by you and answered by the user:

### Q and A ###
"""
    qa = []
    for i, pair in enumerate(qa_pairs):
        qa.append(f"""### Your Question {i + 1} ###
        {pair.system_question}
        ### User's Answer ###
        {pair.user_answer}
        """)

    prompt += "\n".join(qa) + """
### ###

Based on this interaction, please extract new knowledge about the thesis topic and update the existing knowledge.
If the user's query and the response provide new information about the thesis topic, add it to the knowledge.
If the description needs to be refined, please update it.

In your output, you shoudl include both existing knowledge points that are still relevant and new knowledge points extracted from this interaction.
"""
    
    return prompt
