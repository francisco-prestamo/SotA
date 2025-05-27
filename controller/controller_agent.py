from controller.interfaces.agent_interaction_response import AgentInteractionResponse
from controller.interfaces.user_interaction_response import UserInteractionResponse
from controller.interfaces.expert_agent import ExpertAgent
from controller.interfaces.doc_recoverer import DocRecoverer
from controller.interfaces.text_generator import TextGenerator
from controller.interfaces.json_generator import JsonGenerator
from entities.sota_table import SotaTable
from pydantic import BaseModel

class ExpertAgentsNeededModel(BaseModel):
    more_experts_needed: bool
    expert_name_and_descriptions: list[tuple[str, str]] = []
    delete_expert_names: list[str] = []

class SearchNeededModel(BaseModel):
    search_needed: bool

class SuggestionModel(BaseModel):
    suggestion: str

class QueryAndRecovererModel(BaseModel):
    query: str
    selected_index: int

class ControllerAgent:
    def __init__(self, text_generator: TextGenerator, json_generator: JsonGenerator, doc_recoverers: list[DocRecoverer]):
        """
        Initialize the ControllerAgent.
        This class is responsible for coordinating the document embedding process.
        """
        self.text_generator: TextGenerator = text_generator
        self.json_generator: JsonGenerator = json_generator
        self.doc_recoverers: list[DocRecoverer] = doc_recoverers
        self.sota_table: SotaTable = None
        self.experts_agents: list[ExpertAgent] = []

    def user_interact(self, user_interaction: UserInteractionResponse) -> AgentInteractionResponse:
        """
        Orchestrate the interaction between the user and the whole system.
        """
        query = user_interaction.query_text

        if not query:
            return AgentInteractionResponse(text="No query provided.")
    
        expert_prompt = (
            "Given the user query, determine if more expert agents are needed to address the thesis described by the user. "
            "If more experts are needed, provide a list of (name, description) for the required experts. "
            "You can also suggest names of experts to delete if they are no longer relevant. "
            "Respond with a JSON: { 'more_experts_needed': bool, 'expert_name_and_descriptions': [(str,str)], 'delete_expert_names': [str] }\n"
            f"User Query: {query}\n"
            f"Current Expert Agents: {[(e.name,e.description) for e in self.experts_agents]}"
        )

        expert_response: ExpertAgentsNeededModel = self.json_generator.generate_json(expert_prompt, ExpertAgentsNeededModel)

        if hasattr(expert_response, 'delete_expert_names') and expert_response.delete_expert_names:
            self.experts_agents = [e for e in self.experts_agents if e.name not in expert_response.delete_expert_names]

        if expert_response.more_experts_needed and expert_response.expert_name_and_descriptions:
            for name, desc in expert_response.expert_name_and_descriptions:
                new_expert = ExpertAgent(name=name, description=desc)
                self.experts_agents.append(new_expert)

        recoverer_descriptions = [r.description for r in self.doc_recoverers]
        bdi_prompt = (
            "Given the user query, the current SOTA table, and the following document recoverers (with descriptions), "
            "should the system search for new documents? Respond with a JSON: { 'search_needed': bool }\n"
            f"Query: {query}\n"
            f"Current SOTA Table: {self.sota_table}\n"
            f"Recoverers: {recoverer_descriptions}"
        )

        search_needed: SearchNeededModel = self.json_generator.generate_json(bdi_prompt, SearchNeededModel)

        documents = set()
        if search_needed.search_needed:
            expert_suggestions: list[str] = []
            for expert in self.experts_agents:
                expert_prompt = (
                    "Given the following expert's description, the user query, and the current SOTA table, "
                    "suggest what this expert would recommend searching for next. "
                    "Respond with a short suggestion string.\n"
                    f"Expert Description: {expert.description}\n"
                    f"User Query: {query}\n"
                    f"Current SOTA Table: {self.sota_table}"
                )

                suggestion_model = self.json_generator.generate_json(expert_prompt, SuggestionModel)

                if suggestion_model.suggestion:
                    expert_suggestions.append(suggestion_model.suggestion)

            combined_prompt = (
                "Given the user query, the current SOTA table, the following expert suggestions, and the available recoverers with descriptions, "
                "generate the best possible search query for document recovery and select the best recoverer for the query. "
                "Respond with a JSON: { 'query': str, 'selected_index': int }\n"
                f"User Query: {query}\n"
                f"Current SOTA Table: {self.sota_table}\n"
                f"Experts Suggestions: {', '.join(expert_suggestions)}\n"
                f"Recoverers: {', '.join((i, self.doc_recoverers[i].name, self.doc_recoverers[i].description) for i in range(len(self.doc_recoverers)))}\n"
            )

            result = self.json_generator.generate_json(combined_prompt, QueryAndRecovererModel)

            improved_query = result.query
            selected_recoverer = self.doc_recoverers[result.selected_index]
            docs = selected_recoverer.recover(improved_query)
            documents.update(docs)


