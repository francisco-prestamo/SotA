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

from expert_set.interfaces import KnowledgeRecoverer as ExpertSetKR
from receptionist_agent.interfaces import KnowledgeRecoverer as ReceptionistKR



class RecovererAgent(ReceptionistKR, ExpertSetKR):
    def __init__(self, json_generator: JsonGenerator, graphrag: GraphRag, scrappers: List[DocRecoverer], knowledge_graph: KnowledgeGraph):
        self.json_generator: JsonGenerator = json_generator
        self.graphrag: GraphRag = graphrag
        self.scrappers: List[DocRecoverer] = scrappers
        self.kg = knowledge_graph

    def recover_docs(self, query: str, k: int) -> List[Document]:
        print("Searching for:")
        print(query)
        for i in range(3):
            # Get the most relevant text units
            response = self.graphrag.respond(query, self.kg, k)
            relevant_text_units = self.graphrag.get_relevant_text_units_distinct_docs(self.kg, response, top_n=k)
            text_units_strs = [tu.text for tu in relevant_text_units]

            prompt = is_necessary_search_prompt(query, text_units_strs)
            result = self.json_generator.generate_json(prompt, BoolAnswerModel)
            print(f"Iteration for {query} ==================================================================================> {i+1}: {result.answer}")
            print(f"Reasoning: {result.reasoning}")
            if (len(relevant_text_units) > 0):
                print(text_units_strs[0])
            print("-"*100)
            if result.answer:
                return self.graphrag.find_documents(response, self.kg, k)
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
                print(DynamicScrapperSelectionModel.model_json_schema())
                selection_prompt = scrapper_selection_prompt(query, scrapper_infos, DynamicScrapperSelectionModel)
                selection_result = self.json_generator.generate_json(selection_prompt, DynamicScrapperSelectionModel)

                def recover_and_update(s):
                    print(f"Processing scraper: {s.name}")

                    # Check if this scraper was selected
                    scraper_selected = getattr(selection_result, s.name, False)

                    if scraper_selected:
                        # Get the specific query generated for this scraper
                        query_field_name = f"{s.name}_query_to_search"
                        scraper_query = getattr(selection_result, query_field_name, None)

                        # Use the scraper-specific query if available, otherwise fall back to original
                        search_query = scraper_query if scraper_query else query

                        print(f"Using query for {s.name}: {search_query}")

                        # Search using the appropriate query
                        searched_docs = [doc for doc in s.recover(search_query, 20) if getattr(doc, 'content', None)]
                        print(f"Found {len(searched_docs)} documents using {s.name}")

                        for doc in searched_docs:
                            print(f"Found document: {doc.title}")

                        # Update knowledge graph with found documents
                        self.graphrag.update_knowledge_graph(self.kg, searched_docs)
                    else:
                        print(f"Scraper {s.name} was not selected for this research")

                # Process each scraper
                for s in self.scrappers:  # Note: fixed typo from 'scrappers' to 'scrapers'
                    recover_and_update(s)

        response = self.graphrag.respond(query, self.kg, k)
        return self.graphrag.find_documents(response, self.kg, k)

    def get_survey_docs(self, query: str, k=3) -> List[Document]:
        """
        Get documents related to a survey based on the query
        
        Args:
            query: The query to search for survey-related documents
            k: The number of documents to retrieve
            
        Returns:
            A list of retrieved documents from the first scrapper (Semantic Scholar Scrapper)
        """
        print("Searching surveys about:")
        print(query)
        result = list(self.scrappers[0].recover(query, k))
        for doc in result:
            print(f"Found document: {doc.title}")
            print(doc.content[:100])
            print("#"*30)
        return result







