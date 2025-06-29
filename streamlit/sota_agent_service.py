from typing import List, Dict, Tuple

class SotAAgentService:
    """
    Simulated service interface to a multiagent system for generating
    state-of-the-art reviews based on user-provided research topics.
    """

    def __init__(self):
        self.conversation_history: List[Tuple[str, str]] = []
        self.thesis_topic: str = ""
        self.answers: Dict[str, str] = {}

    def get_initial_prompt(self) -> str:
        msg = "Hello! Please provide a thesis topic or article description to begin building your state-of-the-art review."
        self.conversation_history.append(("Recepcionist", msg))
        return msg

    def get_follow_up_questions(self, thesis_topic: str) -> List[str]:
        self.thesis_topic = thesis_topic
        questions = [
            "What specific subfield are you focusing on?",
            "Is your interest more theoretical or applied?",
            "What time period should we consider for recent works?"
        ]
        for q in questions:
            self.conversation_history.append(("Recepcionist", q))
        return questions

    def submit_clarification_answers(self, answers: Dict[str, str]) -> str:
        self.answers.update(answers)
        for question, answer in answers.items():
            self.conversation_history.append(("User", f"{question} -> {answer}"))

        msg = "Thank you! Generating expert profiles and retrieving relevant documents..."
        self.conversation_history.append(("Recepcionist", msg))
        return msg

    def get_state_of_the_art_table(self) -> str:
        table_md = """
| Document         | Methodology         | Dataset       | Key Findings                        |
|------------------|---------------------|----------------|-------------------------------------|
| GNN-Paper-2023    | Graph Neural Nets   | PDBBind        | Achieves SOTA in protein folding    |
| CNN-BioPaper-2022 | Convolutional Nets  | BioCADD        | Outperforms baseline in diagnosis   |
"""
        self.conversation_history.append(("Expert Panel", "Here is the current state-of-the-art table:"))
        self.conversation_history.append(("Expert Panel", table_md))
        return table_md

    def ask_for_clarification_about_table(self) -> str:
        msg = "Should we include datasets from 2024 in the review?"
        self.conversation_history.append(("Expert Panel", msg))
        return msg

    def user_response_to_expert(self, question: str, answer: str) -> str:
        self.conversation_history.append(("User", f"{question} -> {answer}"))
        return "Thanks! Updating the table with your input..."

    def get_chat_history(self) -> List[Tuple[str, str]]:
        return self.conversation_history
