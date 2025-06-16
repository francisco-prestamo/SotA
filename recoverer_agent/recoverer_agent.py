from typing import List
from pydantic import  create_model
from concurrent.futures import ThreadPoolExecutor

from entities.document import Document
from graphrag.knowledge_graph import KnowledgeGraph
from graphrag.graphrag import GraphRag
from recoverer_agent.interfaces.doc_recoverer import DocRecoverer
from recoverer_agent.interfaces.json_generator import JsonGenerator
from recoverer_agent.models.bool_answer_model import BoolAnswerModel
from recoverer_agent.prompts.is_necessary_search_prompt import is_necessary_search_prompt



class RecovererAgent:
    def __init__(self, json_generator: JsonGenerator, graphrag: GraphRag, scrappers: List[DocRecoverer]):
        self.json_generator: JsonGenerator = json_generator
        self.graphrag: GraphRag = graphrag
        self.scrappers: List[DocRecoverer] = scrappers

    def recover_docs(self, query: str, kg: KnowledgeGraph, k: int) -> List[Document]:
        for i in range(3):
            response = self.graphrag.respond(query, kg, k)

            prompt = is_necessary_search_prompt(query, response, BoolAnswerModel)
            result = self.json_generator.generate_json(prompt, BoolAnswerModel)

            if result.answer:
                return self.graphrag.find_documents(response, kg, k)
            else:
                scrapper_infos = [
                    {"name": s.name, "description": s.description}
                    for s in self.scrappers
                ]
                from recoverer_agent.prompts.scrapper_selection_prompt import scrapper_selection_prompt
                
                scrapper_fields = {}
                for s in scrapper_infos:
                    scrapper_fields[s['name']] = (bool, ...)
                    scrapper_fields[f"{s['name']}_query_to_search"] = (str, None)
                DynamicScrapperSelectionModel = create_model(
                    'DynamicScrapperSelectionModel',
                    reasoning=(str, ...),
                    **scrapper_fields
                )
                print("DynamicScrapperSelectionModel schema:")
                print(DynamicScrapperSelectionModel.schema())
                selection_prompt = scrapper_selection_prompt(query, scrapper_infos, DynamicScrapperSelectionModel)
                selection_result = self.json_generator.generate_json(selection_prompt, DynamicScrapperSelectionModel)

                def recover_and_update(s):
                    if getattr(selection_result, s.name, False):
                        searched_docs = s.recover(query)
                        self.graphrag.update_knowledge_graph(kg, searched_docs)
                with ThreadPoolExecutor(max_workers=4) as executor:
                    executor.map(recover_and_update, self.scrappers)

        response = self.graphrag.respond(query, kg, k)
        return self.graphrag.find_documents(response, kg, k)

    def get_survey_docs(self, query: str, k=3) -> List[Document]:
        """
        Get documents related to a survey based on the query
        
        Args:
            query: The query to search for survey-related documents
            k: The number of documents to retrieve
            
        Returns:
            A list of retrieved documents from the first scrapper (Semantic Scholar Scrapper)
        """
        return self.scrappers[0].recover(query)[:k]







