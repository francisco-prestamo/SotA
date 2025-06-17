from typing import List
from board.board import Board, ThesisKnowledgeModel
from expert_set.models.build_expert_model import BuildExpertCommand
from recoverer_agent.recoverer_agent import RecovererAgent
from receptionist_agent.interfaces.json_generator import JsonGenerator
from receptionist_agent.models import ThesisAssessmentModel, BuildExpertCommandList
from receptionist_agent.prompts.thesis_assessment_prompt import thesis_assessment_prompt
from receptionist_agent.prompts.experts_list_prompt import experts_list_prompt
from receptionist_agent.prompts.update_thesis_knowledge_prompt import update_thesis_knowledge_prompt


class ReceptionistAgent:
    """
    The ReceptionistAgent is responsible for interacting with the user to gather knowledge about a thesis topic,
    updating the knowledge in the board, and determining when sufficient knowledge has been gathered.
    """
    
    def __init__(self, 
                 json_generator: JsonGenerator, 
                 board: Board,
                 recoverer_agent: RecovererAgent):
        """
        Initialize the ReceptionistAgent.
        
        Args:
            json_generator: Generator for structured JSON outputs
            board: The central board where knowledge is stored
            recoverer_agent: Agent for recovering documents
        """
        self.json_generator = json_generator
        self.board = board
        self.recoverer_agent = recoverer_agent
        
    def _update_thesis_knowledge(self, user_query: str, response: str) -> ThesisKnowledgeModel:
        """
        Update the thesis knowledge based on the user query and response.
        
        Args:
            user_query: Query from the user
            response: Response to the user's query
            
        Returns:
            Updated thesis knowledge model
        """
        prompt = update_thesis_knowledge_prompt(
            self.board.thesis_knowledge, 
            user_query, 
            response, 
            ThesisKnowledgeModel
        )
        
        updated_knowledge = self.json_generator.generate_json(prompt, ThesisKnowledgeModel)
        self.board.thesis_knowledge = updated_knowledge
        return updated_knowledge
    
    def _is_knowledge_sufficient(self) -> ThesisAssessmentModel:
        """
        Determine if the current knowledge about the thesis topic is sufficient.
        
        Returns:
            Assessment model with the determination
        """
        prompt = thesis_assessment_prompt(self.board.thesis_knowledge, ThesisAssessmentModel)
        assessment = self.json_generator.generate_json(prompt, ThesisAssessmentModel)
        return assessment
    
    def _generate_experts_list(self) -> BuildExpertCommandList:
        """
        Generate a list of experts based on the thesis knowledge.
        
        Returns:
            Model containing a list of experts
        """
        prompt = experts_list_prompt(self.board.thesis_knowledge, BuildExpertCommandList)
        experts_list = self.json_generator.generate_json(prompt, BuildExpertCommandList)
        return experts_list
    
    def interact(self) -> List[BuildExpertCommand]:
        """
        Main interaction loop with the user to gather thesis knowledge.
        """
        print("Welcome to the Thesis State-of-the-Art Assistant!")
        print("I'll help you gather knowledge about your thesis topic and recommend experts.")
        print("Let's start by discussing your thesis topic.")
        
        if not self.board.thesis_knowledge.description:
            user_input = input("What is your thesis topic about? ")
            # Initialize thesis knowledge with first input
            self.board.thesis_knowledge = ThesisKnowledgeModel(
                thoughts=[],
                description=user_input
            )
            
        knowledge_sufficient = False
        
        while not knowledge_sufficient:
            # Check if knowledge is sufficient
            assessment = self._is_knowledge_sufficient()
            knowledge_sufficient = assessment.is_sufficient
            
            if knowledge_sufficient:
                print("\nGreat! I now have sufficient knowledge about your thesis topic.")
                break
                
            # If not sufficient, ask more questions
            print("\nI need to know more about your thesis to provide the best recommendations.")
            if assessment.missing_aspects:
                print("I'm missing information about these aspects:")
                for aspect in assessment.missing_aspects:
                    print(f"- {aspect}")
            
            # Suggest questions or let user ask freely
            if assessment.suggested_questions:
                print("\nHere are some questions I could help with:")
                for i, question in enumerate(assessment.suggested_questions, 1):
                    print(f"{i}. {question}")
                print("0. Ask your own question")
                
                choice = input("\nEnter the number of a question or 0 to ask your own: ")
                if choice.isdigit() and 1 <= int(choice) <= len(assessment.suggested_questions):
                    user_query = assessment.suggested_questions[int(choice)-1]
                else:
                    user_query = input("\nWhat would you like to know about your thesis topic? ")
            else:
                user_query = input("\nWhat would you like to know about your thesis topic? ")
            
            # Get response using the recoverer agent
            docs = self.recoverer_agent.recover_docs(user_query, self.board.knowledge_graph, 3)
            
            # Format a response based on the retrieved documents
            if docs:
                response = "Based on the information I've found:\n"
                for i, doc in enumerate(docs, 1):
                    response += f"\nSource {i}: {doc.title}\n"
                    response += f"Abstract: {doc.abstract}\n"
                    if len(doc.content) > 300:
                        response += f"Content (excerpt): {doc.content[:300]}...\n"
                    else:
                        response += f"Content: {doc.content}\n"
            else:
                response = "I couldn't find specific information about that. Could you provide more details or rephrase your question?"
            
            print("\n" + response)
            
            # Update thesis knowledge
            self._update_thesis_knowledge(user_query, response)
        
        # Generate experts list when knowledge is sufficient
        experts_list = self._generate_experts_list()
        
        # Display experts
        print("\n--- Recommended Experts ---")
        for i, expert in enumerate(experts_list.experts, 1):
            print(f"\nExpert {i}: {expert.name}")
            print(f"Expertise: {expert.description}")
            print(f"Recommended search query: '{expert.query}'")
        
        print("\nThank you for using the Thesis State-of-the-Art Assistant!")
        
        return experts_list.experts
