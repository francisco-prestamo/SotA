from typing import List
from board.board import Board, ThesisKnowledgeModel
from expert_set.models.build_expert_model import BuildExpertCommand
from recoverer_agent.recoverer_agent import RecovererAgent
from receptionist_agent.interfaces import JsonGenerator, UserAPI
from receptionist_agent.models import ThesisAssessmentModel, BuildExpertCommandList
from receptionist_agent.prompts.thesis_assessment_prompt import thesis_assessment_prompt
from receptionist_agent.prompts.experts_list_prompt import experts_list_prompt
from receptionist_agent.prompts.update_thesis_knowledge_prompt import (
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
        recoverer_agent: RecovererAgent,
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

    def _update_thesis_knowledge(
        self, user_query: str, response: str
    ) -> ThesisKnowledgeModel:
        """
        Update the thesis knowledge based on the user query and response.

        Args:
            user_query: Query from the user
            response: Response to the user's query

        Returns:
            Updated thesis knowledge model
        """
        prompt = update_thesis_knowledge_prompt(
            self.board.thesis_knowledge, user_query, response, ThesisKnowledgeModel
        )

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
        prompt = thesis_assessment_prompt(self.board.thesis_knowledge)
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
        Main interaction loop with the user to gather thesis knowledge.
        """
        query_parts = [
            "Welcome to the Thesis State-of-the-Art Assistant!",
            "I'll help you gather knowledge about your thesis topic and recommend experts.",
            "Let's start with a simple question: What is your thesis topic?",
        ]
        self.user_api.message_user("\n".join(query_parts))

        if not self.board.thesis_knowledge.description:
            user_input = self.user_api.query_user("Please describe your thesis topic: ")
            if not user_input.strip():
                self.user_api.message_user("No input received. Please provide a description to continue.")
                return []
            self.board.thesis_knowledge = ThesisKnowledgeModel(
                thoughts=[], description=user_input
            )

        while True:
            assessment = self._is_knowledge_sufficient()
            if assessment.is_sufficient:
                self.user_api.message_user("Great! I now have enough information about your thesis topic.")
                break

            followup = [
                "To help you better, I need a bit more information about your thesis."
            ]
            if assessment.missing_aspects:
                followup.append("Could you tell me more about these aspects?")
                for aspect in assessment.missing_aspects:
                    followup.append(f"- {aspect}")
            self.user_api.message_user("\n".join(followup))

            qa_pairs = []
            if assessment.suggested_questions:
                for question in assessment.suggested_questions:
                    user_answer = self.user_api.query_user(f"{question} ")
                    if not user_answer.strip():
                        self.user_api.message_user("No input received. Please respond to continue.")
                        continue
                    # Optionally, you could use recoverer_agent here if needed
                    qa_pairs.append((question, user_answer))
            else:
                user_query = self.user_api.query_user(
                    "What would you like to add or clarify about your thesis topic? "
                )
                if not user_query.strip():
                    self.user_api.message_user("No input received. Please respond to continue.")
                    continue
                qa_pairs.append((user_query, user_query))

            # Update thesis knowledge with all Q&A pairs
            for user_query, user_response in qa_pairs:
                self._update_thesis_knowledge(user_query, user_response)

        experts_list = self._generate_experts_list()
        self.user_api.message_user("\n--- Recommended Experts ---")
        for i, expert in enumerate(experts_list.experts, 1):
            self.user_api.message_user(f"\nExpert {i}: {expert.name}\nExpertise: {expert.description}\nRecommended search query: '{expert.query}'")
        self.user_api.message_user("\nThank you for using the Thesis State-of-the-Art Assistant!")
        return experts_list.experts
