from controller.interfaces.docset_builder import DocsetBuilder
from rar_engine.interfaces.doc_recoverer import DocRecoverer
from rar_engine.interfaces.text_text_llm import TextGenerator
from rar_engine.interfaces.json_generator import JsonGenerator
from entities.document import Document
from typing import List, Any, Dict
from pydantic import BaseModel, ValidationError

class Beliefs(BaseModel):
    thoughts: List[str] = []
    already_searched: List[str] = []

class MergeQueryModel(BaseModel):
    query: str

class DesiresModel(BaseModel):
    desires: list[str]

class IntentionsModel(BaseModel):
    intentions: list[str]

class RecovererSelectionModel(BaseModel):
    name: str


class BDIAgent(DocsetBuilder):
    def __init__(self, recoverers: List[DocRecoverer], text_llm: TextGenerator, json_llm: JsonGenerator):
        """
        recoverers: list of available document recoverers (must implement DocRecoverer interface)
        text_llm: LLM implementing TextGenerator for generating questions and explanations
        json_llm: LLM implementing JsonGenerator for structured belief updates
        """
        self.recoverers: List[DocRecoverer] = recoverers
        self.text_llm: TextGenerator = text_llm
        self.json_llm: JsonGenerator = json_llm
        self.beliefs: Beliefs = Beliefs()
        self.desires: List[str] = []
        self.intentions: List[str] = []
        self.user_query: str = None
        self.docs: List[Document] = []
        self.clarified: bool = False
        self.history: List[dict] = []

    def ask_questions(self):
        """Use LLM to generate clarifying questions and update beliefs, max 3 iterations."""
        max_iterations = 3
        iteration = 0
        current_query = self.user_query
        while not self.clarified and iteration < max_iterations:
            prompt = (
                "Given the following user query, generate a clarifying question if more information is needed. "
                "If enough information is present, respond with 'NO_QUESTION_NEEDED'.\n"
                f"User query: {current_query}\n"
            )
            question = self.text_llm.generate_text(prompt)
            if question.strip() == 'NO_QUESTION_NEEDED':
                self.clarified = True
                self.user_query = current_query
                break

            # Ask the user the question and get an answer (placeholder for now)
            answer = "This is a placeholder answer."

            self.history.append({"role": "agent", "content": question})
            self.history.append({"role": "user", "content": answer})

            merge_prompt = (
                "Given the original user query and the following new information, update the query to incorporate both. "
                "Respond with the updated query as a string.\n"
                f"Original query: {current_query}\n"
                f"New information: {answer}\n"
            )
            updated_query = self.json_llm.generate_json(merge_prompt, MergeQueryModel).query
            current_query = updated_query if updated_query else current_query
            iteration += 1

        self.user_query = current_query

    def generate_desires(self):
        """Use LLM to generate desires/goals from beliefs."""
        prompt = (
            "Given the current beliefs, list the agent's desires of geting diferents types of papers in diferents sites\n"
            f"Beliefs: {self.beliefs}\n"
        )
        desires_obj = self.json_llm.generate_json(prompt, DesiresModel)
        self.desires = desires_obj.desires
        

    def filter_desires(self):
        """Use LLM to filter/prioritize desires into intentions."""
        prompt = (
            "Given the current desires and beliefs, select the intentions to pursue as a JSON list of strings.\n"
            f"Desires: {self.desires}\nBeliefs: {self.beliefs}\n"
        )
        intentions_obj = self.json_llm.generate_json(prompt, IntentionsModel)
        self.intentions = intentions_obj.intentions
        

    def select_recoverer(self) -> DocRecoverer:
        """Use LLM and JSON generator to select the best recoverer."""
        recoverer_names = [r.name for r in self.recoverers]
        prompt = (
            "Given the intentions and beliefs, select the best recoverer from the following list. "
            "Respond with a JSON object: {\"name\": <recoverer_name>} where <recoverer_name> must be one of: "
            f"{recoverer_names}.\n"
            f"Intentions: {self.intentions}\nBeliefs: {self.beliefs}\n"
        )
        result = self.json_llm.generate_json(prompt, RecovererSelectionModel)
        chosen_name = result.name
        
        for r in self.recoverers:
            if r.name == chosen_name:
                return r
            
        return self.recoverers[0]

    def generate_recoverer_query(self) -> str:
        """Use LLM to generate a query for the recoverer."""
        prompt = (
            "Given the beliefs and intentions, generate the best query string for the recoverer.\n"
            f"Beliefs: {self.beliefs}\nIntentions: {self.intentions}\n"
        )
        return self.text_llm.generate_text(prompt)

    def inform_llm_of_results(self, searched: str, recoverer_name: str, docs):
        """Inform the LLM about the results and update beliefs."""
        class BeliefsUpdateModel(Beliefs):
            pass
        prompt = (
            "Given the current beliefs and the following recovered documents, update the beliefs as JSON that matches this schema.\n"
            f"Current beliefs: {self.beliefs}\n"
            f"Recovered docs: {[str(d) for d in docs]}\n"
            f"JSON Schema: {Beliefs.schema_json()}\n"
        )
        new_beliefs = self.json_llm.generate_json(prompt, BeliefsUpdateModel)
        self.update_beliefs(Beliefs.parse_obj(new_beliefs).dict())

    def build_doc_set(self, query: str) -> List[Document]:
        """
        Main method to run the BDI agent.
        :param query: The user query to process.
        :return: The final documents.
        """
        self.user_query = query
        self.clarified = False
        self.history = [
            {"role": "user", "content": query}
        ]

        
        initial_beliefs_prompt = (
            "Given the user query, generate the beliefs that be focused to search for papers in diferents sites related to the user query.\n"
            f"User query: {self.user_query}\n"
            "The beliefs return it as JSON that matches this schema.\n"
            f"JSON Schema: {Beliefs.schema_json()}\n"
        )

        self.beliefs = self.json_llm.generate_json(initial_beliefs_prompt, Beliefs)

        if not self.clarified:
            self.ask_questions()
            if not self.clarified:
                print("Could not clarify the query. Aborting.")
                return []
            
        MAX_ITERATIONS = 5
        iteration_count = 0
        
        while iteration_count < MAX_ITERATIONS:
            self.generate_desires()
            self.filter_desires()
            if not self.intentions:
                print("No intentions to pursue. Stopping.")
                break
            recoverer = self.select_recoverer()
            if not recoverer:
                print("No recoverer available. Stopping.")
                break
            query = self.generate_recoverer_query()
            docs = recoverer.recover(query)
            self.docs.extend(docs)
            self.inform_llm_of_results(query,recoverer.name,docs)

            continue_prompt = (
                "Given the current beliefs and the last recovered documents, should the agent continue to iterate? "
                f"Beliefs: {self.beliefs}\nDocs: {[str(d) for d in docs]}\n"
                "Respond with 'yes' or 'no'.\n"
            )
            cont = self.text_llm.generate_text(continue_prompt)
            if cont.strip().lower() != 'yes':
                break
            iteration_count += 1
            if iteration_count >= MAX_ITERATIONS:
                print("Reached maximum iterations.")
                
        return self.docs