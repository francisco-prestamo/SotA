from typing import List, Dict, Set
from pydantic import create_model
from concurrent.futures import ThreadPoolExecutor

from entities.document import Document
from graphrag.knowledge_graph import KnowledgeGraph
from graphrag.graphrag import GraphRag
from recoverer_agent.interfaces.doc_recoverer import DocRecoverer
from recoverer_agent.interfaces.json_generator import JsonGenerator
from recoverer_agent.models.bool_answer_model import BoolAnswerModel
from recoverer_agent.models.scrapper_selection_model import ScrapperSelectionResponse
from recoverer_agent.prompts.is_necessary_search_prompt import is_necessary_search_prompt

from expert_set.interfaces import KnowledgeRecoverer as ExpertSetKR
from receptionist_agent.interfaces import KnowledgeRecoverer as ReceptionistKR



class RecovererAgent(ReceptionistKR, ExpertSetKR):
    def __init__(self, json_generator: JsonGenerator, graphrag: GraphRag, scrappers: List[DocRecoverer], knowledge_graph: KnowledgeGraph):
        self.json_generator: JsonGenerator = json_generator
        self.graphrag: GraphRag = graphrag
        self.scrappers: List[DocRecoverer] = scrappers
        self.kg = knowledge_graph
        self.tracked_searches = {}  # Dictionary to track searches: {recoverer_name: [list of searched queries]}

    def recover_docs(self, query: str, k: int) -> List[Document]:
        print("Searching for:")
        print(query)
        for i in range(2):
            # Get the most relevant text units
            response = self.graphrag.respond(query, self.kg, k)
            relevant_text_units = self.graphrag.get_relevant_text_units_distinct_docs(self.kg, response, top_n=k)
            text_units_strs = [tu.text for tu in relevant_text_units]

            prompt = is_necessary_search_prompt(query, text_units_strs)
            result = self.json_generator.generate_json(prompt, BoolAnswerModel)
            print(f"Iteration ===============> {i+1}: {result.answer}")
            if (len(relevant_text_units) > 0):
                print(text_units_strs[0])
            print("-"*100)
            if result.answer:
                docs = self.graphrag.find_documents(response, self.kg, k)  
                print(f"Found {len(docs)} documents in the knowledge graph.")
                return docs
            else:
                # Add search history to scraper info
                scrapper_infos = [
                    {
                        "name": s.name, 
                        "description": s.description,
                        "previous_searches": self.tracked_searches.get(s.name, [])
                    }
                    for s in self.scrappers
                ]
                from recoverer_agent.prompts.scrapper_selection_prompt import scrapper_selection_prompt
                
                # Use our new model for scraper selection
                selection_prompt = scrapper_selection_prompt(query, scrapper_infos, ScrapperSelectionResponse)
                selection_result = self.json_generator.generate_json(selection_prompt, ScrapperSelectionResponse)
                
                def recover_and_update(s):
                    print(f"Processing scraper: {s.name}")
                    
                    # Find if this scraper was selected in the results
                    scraper_selection = next((sel for sel in selection_result.selections if sel.source_name == s.name), None)
                    
                    if scraper_selection and scraper_selection.selected and scraper_selection.queries:
                        print(f"Scraper {s.name} selected with {len(scraper_selection.queries)} queries")
                        all_docs = []
                        # Removed duplicate date_filter declaration
                        
                        # Process each query for this scraper
                        for query_item in scraper_selection.queries:
                            search_query = query_item.query
                            
                            # Track this search query for this scraper
                            if s.name not in self.tracked_searches:
                                self.tracked_searches[s.name] = []
                            self.tracked_searches[s.name].append(search_query)
                            
                            print(f"Using query for {s.name}: {search_query}")
                            print(f"Query reasoning: {query_item.reasoning}")
                            
                            # Search using the query with date filter
                        # Format: ("YYYY-MM-DD", "YYYY-MM-DD") for start and end dates
                        date_filter = ("2020-01-01", "2025-02-20")  # Adjust time range as needed
                        searched_docs = [doc for doc in s.recover(query=search_query, k=2, date_filter=date_filter)
                                        if getattr(doc, 'content', None)]
                        all_docs.extend(searched_docs)
                        print(f"Found {len(searched_docs)} documents using query: {search_query}")
                        
                        for doc in searched_docs:
                            print(f"Found document: {doc.title}")
                        
                        # Update knowledge graph with all found documents
                        if all_docs:
                            print(f"Found total of {len(all_docs)} documents using {s.name}")
                            self.graphrag.update_knowledge_graph(self.kg, all_docs)
                    else:
                        print(f"Scraper {s.name} was not selected for this research or had no queries")

                # Print the overall reasoning
                print(f"Selection reasoning: {selection_result.reasoning}")
                
                # Process each scraper in parallel for better performance
                with ThreadPoolExecutor() as executor:
                    executor.map(recover_and_update, self.scrappers)

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







