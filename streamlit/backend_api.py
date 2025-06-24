from typing import List, Dict

conversation_history = []

def get_initial_prompt() -> str:
    """
    Returns the initial message from the Recepcionist agent.
    """
    msg = "Hello! Please provide a thesis topic or article description to begin building your state-of-the-art review."
    conversation_history.append(("Recepcionist", msg))
    return msg

def get_follow_up_questions(thesis_topic: str) -> List[str]:
    """
    Returns a list of follow-up questions to clarify the thesis topic.
    """
    questions = [
        "What specific subfield are you focusing on?",
        "Is your interest more theoretical or applied?",
        "What time period should we consider for recent works?"
    ]
    for q in questions:
        conversation_history.append(("Recepcionist", q))
    return questions

def submit_clarification_answers(answers: Dict[str, str]) -> str:
    """
    Accepts answers from the user to the clarification questions.
    Returns confirmation or next step.
    """
    for question, answer in answers.items():
        conversation_history.append(("User", f"{question} -> {answer}"))

    msg = "Thank you! Generating expert profiles and retrieving relevant documents..."
    conversation_history.append(("Recepcionist", msg))
    return msg

def get_state_of_the_art_table() -> str:
    """
    Returns a markdown table of the current state of the art.
    """
    table_md = """
| Document         | Methodology         | Dataset       | Key Findings                        |
|------------------|---------------------|----------------|-------------------------------------|
| GNN-Paper-2023    | Graph Neural Nets   | PDBBind        | Achieves SOTA in protein folding    |
| CNN-BioPaper-2022 | Convolutional Nets  | BioCADD        | Outperforms baseline in diagnosis   |
"""
    conversation_history.append(("Expert Panel", "Here is the current state-of-the-art table:"))
    conversation_history.append(("Expert Panel", table_md))
    return table_md

def ask_for_clarification_about_table() -> str:
    """
    Returns a follow-up question from the experts about the state of the art.
    """
    msg = "Should we include datasets from 2024 in the review?"
    conversation_history.append(("Expert Panel", msg))
    return msg

def user_response_to_expert(question: str, answer: str) -> str:
    """
    Records user's response to expert question.
    """
    conversation_history.append(("User", f"{question} -> {answer}"))
    return "Thanks! Updating the table with your input..."

def get_chat_history() -> List[tuple]:
    """
    Returns the entire conversation so far.
    """
    return conversation_history
