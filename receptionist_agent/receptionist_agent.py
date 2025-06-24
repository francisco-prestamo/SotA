from typing import List
from board.board import Board, ThesisKnowledgeModel
from expert_set.models.build_expert_model import BuildExpertCommand
from receptionist_agent.interfaces import JsonGenerator, UserAPI, KnowledgeRecoverer
from receptionist_agent.models import ThesisAssessmentModel, BuildExpertCommandList
from receptionist_agent.prompts.thesis_assessment_prompt import thesis_assessment_prompt
from receptionist_agent.prompts.experts_list_prompt import experts_list_prompt
from receptionist_agent.prompts.update_thesis_knowledge_prompt import (
    QAndA,
    update_thesis_knowledge_prompt,
)


class ReceptionistAgent:
    """
    The ReceptionistAgent is responsible for interacting with the user to gather knowledge about a thesis topic,
    updating the knowledge in the board, and determining when sufficient knowledge has been gathered.
    """

    def __init__(
        self,
        json_generator: JsonGenerator,
        board: Board,
        recoverer_agent: KnowledgeRecoverer,
        user_api: UserAPI,
    ):
        """
        Initialize the ReceptionistAgent.

        Args:
            json_generator: Generator for structured JSON outputs
            board: The central board where knowledge is stored
            recoverer_agent: Agent for recovering documents
            user_api: Interface for querying the user
        """
        self.json_generator = json_generator
        self.board = board
        self.recoverer_agent = recoverer_agent
        self.user_api = user_api
        self.messages = []  # List of chat messages (dicts with sender/content)

    def add_message(self, sender: str, content: str):
        self.messages.append({"sender": sender, "content": content})

    def _update_thesis_knowledge(self, qa_pairs: List[QAndA]) -> ThesisKnowledgeModel:
        """
        Update the thesis knowledge based on the user query and response.

        Args:
            user_query: Query from the user
            response: Response to the user's query

        Returns:
            Updated thesis knowledge model
        """
        prompt = update_thesis_knowledge_prompt(self.board.thesis_knowledge, qa_pairs)

        updated_knowledge = self.json_generator.generate_json(
            prompt, ThesisKnowledgeModel
        )
        self.board.thesis_knowledge = updated_knowledge
        return updated_knowledge

    def _is_knowledge_sufficient(self) -> ThesisAssessmentModel:
        """
        Determine if the current knowledge about the thesis topic is sufficient.

        Returns:
            Assessment model with the determination
        """
        prompt = thesis_assessment_prompt(self.board.thesis_knowledge, self.messages)
        assessment = self.json_generator.generate_json(prompt, ThesisAssessmentModel)
        return assessment

    def _generate_experts_list(self) -> BuildExpertCommandList:
        """
        Generate a list of experts based on the thesis knowledge.

        Returns:
            Model containing a list of experts
        """
        prompt = experts_list_prompt(self.board.thesis_knowledge)
        experts_list = self.json_generator.generate_json(prompt, BuildExpertCommandList)
        return experts_list

    def interact(self) -> List[BuildExpertCommand]:
        """
        Main interaction loop with the user to gather thesis knowledge, using a chat/message model.
        """
        self.messages = []  # Reset chat history for each session
        query_parts = [
            "Welcome to the Thesis State-of-the-Art Assistant!",
            "I'll help you gather knowledge about your thesis topic and recommend experts.",
            "Let's start with a simple question: What is your thesis topic?",
        ]
        welcome_msg = "\n".join(query_parts)
        self.user_api.receptionist_message_user(welcome_msg)
        self.add_message("receptionist", welcome_msg)

        if not self.board.thesis_knowledge.description:
            user_input = self.user_api.receptionist_query_user("Please describe your thesis topic: ")
            self.add_message("user", user_input)
            if not user_input.strip():
                self.user_api.receptionist_message_user(
                    "No input received. Please provide a description to continue."
                )
                self.add_message(
                    "receptionist",
                    "No input received. Please provide a description to continue.",
                )
                return []
            self.board.thesis_knowledge = ThesisKnowledgeModel(
                thoughts=[], description=user_input
            )

        while True:
            # Use chat history and thesis knowledge in the assessment prompt
            assessment_prompt = thesis_assessment_prompt(
                self.board.thesis_knowledge, self.messages
            )
            assessment = self.json_generator.generate_json(
                assessment_prompt, ThesisAssessmentModel
            )
            if assessment.is_sufficient:
                done_msg = (
                    "Great! I now have enough information about your thesis topic."
                )
                self.user_api.receptionist_message_user(done_msg)
                self.add_message("receptionist", done_msg)
                break

            followup = [
                "To help you better, I need a bit more information about your thesis."
            ]
            if assessment.missing_aspects:
                followup.append("Could you tell me more about these aspects?")
                for aspect in assessment.missing_aspects:
                    followup.append(f"- {aspect}")
            followup_msg = "\n".join(followup)
            self.user_api.receptionist_message_user(followup_msg)
            self.add_message("receptionist", followup_msg)

            qa_pairs = []
            if assessment.suggested_questions:
                for question in assessment.suggested_questions:
                    user_answer = self.user_api.receptionist_query_user(f"{question} ")
                    self.add_message("receptionist", question)
                    self.add_message("user", user_answer)
                    if not user_answer.strip():
                        self.user_api.receptionist_message_user(
                            "No input received. Please respond to continue."
                        )
                        self.add_message(
                            "receptionist",
                            "No input received. Please respond to continue.",
                        )
                        continue
                    qa_pairs.append(
                        QAndA(system_question=question, user_answer=user_answer)
                    )
            else:
                question = (
                    "What would you like to add or clarify about your thesis topic? "
                )
                user_answer = self.user_api.receptionist_query_user(question)
                self.add_message(
                    "receptionist",
                    question,
                )
                self.add_message("user", user_answer)
                if not user_answer.strip():
                    self.user_api.receptionist_message_user(
                        "No input received. Please respond to continue."
                    )
                    self.add_message(
                        "receptionist", "No input received."
                    )
                    continue
                qa_pairs.append(QAndA(system_question=question, user_answer=user_answer))

            # Update thesis knowledge with all Q&A pairs
            self._update_thesis_knowledge(qa_pairs)

        experts_list = self._generate_experts_list()
        experts_msg = "\n--- Recommended Experts ---"
        self.user_api.receptionist_message_user(experts_msg)
        self.add_message("receptionist", experts_msg)
        for i, expert in enumerate(experts_list.experts, 1):
            expert_msg = f"\nExpert {i}: {expert.name}\nExpertise: {expert.description}\nRecommended search query: '{expert.query}'"
            self.user_api.receptionist_message_user(expert_msg)
            self.add_message("receptionist", expert_msg)
        thanks_msg = "\nThank you for using the Thesis State-of-the-Art Assistant!"
        self.user_api.receptionist_message_user(thanks_msg)
        self.add_message("receptionist", thanks_msg)
        return experts_list.experts
