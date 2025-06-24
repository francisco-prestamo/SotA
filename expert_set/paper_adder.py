from typing import List, Dict
import logging
from pydantic import create_model

from entities.sota_table import sota_table_to_markdown, PaperFeaturesModel
from expert_set.models.expert import Expert

from .interfaces import JsonGenerator
from recoverer_agent import RecovererAgent
from board.board import Board
from entities.document import Document
from .models.expert_search_reasoning_model import build_expert_search_reasoning_model
from .prompts.build_expert_search_reasoning_prompt import build_expert_search_reasoning_prompt

from .models.paper_addition_result_model import PaperAdditionResult
from .models.expert_chunk_new_features_model import ExpertChunkNewFeatures
from .models.paper_feature_extraction_model import PaperFeatureExtraction
from .models.string_response_model import StringResponseModel
from .models.dict_response_model import DictResponseModel

from .prompts.feature_extraction_prompt import build_feature_extraction_prompt
from .prompts.new_feature_identification_prompt import build_new_feature_identification_prompt
from .prompts.feature_consolidation_prompt import build_feature_consolidation_prompt
from .prompts.domain_extraction_prompt import build_domain_extraction_prompt
from .prompts.addition_summary_prompt import build_addition_summary_prompt
from .prompts.search_query_synthesis_prompt import build_search_query_synthesis_prompt


def create_features_extraction_model(feature_list: List[str]):
    """Create a dynamic model with all features as fields"""
    # Create fields for each feature
    fields = {feature: (str, "") for feature in feature_list}
    
    return create_model('FeaturesExtractionModel', **fields)


class PaperAdder:
    def __init__(
        self,
        json_generator: JsonGenerator,
        board: Board,
        recoverer_agent: RecovererAgent,
        k: int = 5
    ):
        self.json_generator = json_generator
        self.recoverer_agent = recoverer_agent
        self.k = k
        self.board = board

    def add_papers(self, experts: List[Expert]) -> PaperAdditionResult:
        """
        For each expert, use the LLM to reason about what is missing from the SOTA table,
        search for relevant documents, and add them to the table.
        """
        # Get expert search reasoning
        print("Adding papers to SOTA table...")
        expert_names = [expert.name for expert in experts]
        expert_search_reasoning_model = build_expert_search_reasoning_model(expert_names)
        
        # Build context for reasoning
        sota_md = sota_table_to_markdown(self.board.sota_table)
        thesis_desc = self.board.thesis_knowledge.description
        thesis_thoughts = self.board.thesis_knowledge.thoughts
        expert_context = {
            expert.name: {"expert_description": expert.expert_model.description}
            for expert in experts
        }
        
        # Get search reasoning from experts
        prompt = build_expert_search_reasoning_prompt(sota_md, thesis_desc, thesis_thoughts, expert_context)
        expert_searches = self.json_generator.generate_json(prompt, expert_search_reasoning_model)
        
        # Synthesize search queries
        all_queries = [getattr(expert_searches, name).what_to_search for name in expert_names]
        synthesis_prompt = build_search_query_synthesis_prompt(all_queries)
        summary_query_model = self.json_generator.generate_json(synthesis_prompt, StringResponseModel)
        summary_query = summary_query_model.response
        print(f"Search query: {summary_query}")
        # Recover new documents
        new_docs = self.recoverer_agent.recover_docs(summary_query, self.board.knowledge_graph, self.k)

        if not new_docs:
            logging.warning("No new documents were recovered.")
            return PaperAdditionResult(papers_added=[], summary="No new documents were recovered.")
        
        # Add each document to the SOTA table
        added_titles = []
        original_features = set(self.board.sota_table.features)
        
        for doc in new_docs:
            print(f"Adding {doc.title} to SOTA table...")
            self._add_paper_to_sota_table(doc, experts)
            added_titles.append(doc.title)
        
        # Identify new features that were added
        new_features = [f for f in self.board.sota_table.features if f not in original_features]
        
        # Generate summary
        summary_prompt = build_addition_summary_prompt(
            added_titles, expert_names, new_features
        )
        summary_model = self.json_generator.generate_json(summary_prompt, StringResponseModel)
        summary = summary_model.response
        
        return PaperAdditionResult(papers_added=added_titles, summary=summary)

    def _add_paper_to_sota_table(self, doc: Document, experts: List[Expert]) -> None:
        """Add a single paper to the SOTA table by extracting features"""
        # Extract features from the document
        extraction_result = self._extract_paper_features(doc, experts)
         # Update SOTA table features if new ones were found
        new_features_added = []
        for new_feature in extraction_result.consolidated_new_features:
            if new_feature not in self.board.sota_table.features:
                self.board.sota_table.features.append(new_feature)
                new_features_added.append(new_feature)
        
        # If we added new features, process all existing documents to check for these features
        if new_features_added:
            from expert_set.utils.document_chunking import chunk_document
            
            for existing_doc, existing_features in self.board.sota_table.document_features:
                # First initialize with "Not Available"
                for new_feature in new_features_added:
                    if new_feature not in existing_features.features:
                        existing_features.features[new_feature] = {"value": "Not Available"}
                
                # Now process document chunks to find values for the new features
                chunks = chunk_document(existing_doc)
                chunk_new_feature_values = []
                
                # Process each chunk to find new features
                for chunk_idx, chunk in enumerate(chunks):
                    # Extract values for the new features from this chunk
                    feature_values = self._extract_new_feature_values_from_chunk(
                        chunk, chunk_idx, new_features_added, existing_doc
                    )
                    if feature_values:
                        chunk_new_feature_values.append(feature_values)
                
                # Consolidate the feature values found across all chunks
                if chunk_new_feature_values:
                    for new_feature in new_features_added:
                        # Collect all values for this feature from different chunks
                        values = []
                        for feature_dict in chunk_new_feature_values:
                            if new_feature in feature_dict and feature_dict[new_feature]:
                                values.append(feature_dict[new_feature])
                        
                        # If we found values, consolidate them
                        if values:
                            prompt = build_feature_consolidation_prompt(
                                new_feature, existing_doc.title, values
                            )
                            try:
                                consolidated_model = self.json_generator.generate_json(prompt, StringResponseModel)
                                consolidated_value = consolidated_model.response
                                existing_features.features[new_feature] = {"value": consolidated_value}
                            except Exception as e:
                                logging.warning(f"Failed to consolidate feature {new_feature} for document {existing_doc.title}: {e}")
        
        # Combine all features for this paper
        all_features = {}
        
        # Add existing features
        for feature in self.board.sota_table.features:
            if feature in extraction_result.consolidated_features:
                all_features[feature] = {"value": extraction_result.consolidated_features[feature]}
            elif feature in extraction_result.consolidated_new_features:
                all_features[feature] = {"value": extraction_result.consolidated_new_features[feature]}
            else:
                all_features[feature] = {"value": "Not Available"}
        
        # Create PaperFeaturesModel
        paper_features = PaperFeaturesModel(
            authors=doc.authors,
            title=doc.title,
            year=self._extract_year_from_id(doc.id),
            domain=self._extract_domain(doc, experts),
            features=all_features
        )
        
        # Add to SOTA table
        self.board.sota_table.document_features.append((doc, paper_features))

    def _extract_paper_features(self, doc: Document, experts: List[Expert]) -> PaperFeatureExtraction:
        """Extract features from a paper using multiple experts across chunks"""
        from expert_set.utils.document_chunking import chunk_document
        chunks = chunk_document(doc)
        
        chunk_features = []
        chunk_new_features = []
        
        # Process each chunk with each expert
        for chunk_idx, chunk in enumerate(chunks):
            for expert in experts:
                # Extract existing features
                existing_features = self._extract_features_from_chunk(
                    expert, doc, chunk, chunk_idx
                )
                chunk_features.append(existing_features)
                
                # Identify new features
                new_features = self._identify_new_features_from_chunk(
                    expert, doc, chunk, chunk_idx
                )
                chunk_new_features.append(new_features)
        
        # Consolidate features across all chunks and experts
        consolidated_features = self._consolidate_features(chunk_features, doc.title)
        consolidated_new_features = self._consolidate_new_features(chunk_new_features, doc.title)
        
        return PaperFeatureExtraction(
            document=doc,
            chunk_features=chunk_features,
            chunk_new_features=chunk_new_features,
            consolidated_features=consolidated_features,
            consolidated_new_features=consolidated_new_features
        )

    def _extract_features_from_chunk(
        self, 
        expert: Expert, 
        doc: Document, 
        chunk, 
        chunk_idx: int
    ) -> Dict[str, str]:
        """Extract existing SOTA table features from a chunk using an expert"""
        prompt = build_feature_extraction_prompt(
            expert.expert_model.description,
            self.board.thesis_knowledge.description,
            doc.title,
            doc.authors,
            chunk.chunk,
            self.board.sota_table.features
        )
        
        # Create dynamic model with all features
        FeaturesModel = create_features_extraction_model(self.board.sota_table.features)
        
        try:
            response = self.json_generator.generate_json(prompt, FeaturesModel)
            # Extract all features from the model instance
            extracted_features = {}
            for feature in self.board.sota_table.features:
                extracted_features[feature] = getattr(response, feature, "")
            
        except Exception as e:
            logging.warning(f"Failed to extract features for expert {expert.name}, chunk {chunk_idx}: {e}")
            extracted_features = {feature: "" for feature in self.board.sota_table.features}
        
        return extracted_features

    def _identify_new_features_from_chunk(
        self, 
        expert: Expert, 
        doc: Document, 
        chunk, 
        chunk_idx: int
    ) -> ExpertChunkNewFeatures:
        """Identify new features in a chunk using an expert"""
        prompt = build_new_feature_identification_prompt(
            expert.expert_model.description,
            self.board.thesis_knowledge.description,
            doc.title,
            doc.authors,
            chunk.chunk,
            self.board.sota_table.features
        )
        
        try:
            response_model = self.json_generator.generate_json(prompt, DictResponseModel)
            response = response_model.data
            new_features = response.get("new_features", [])
            feature_values = response.get("feature_values", {})
        except Exception as e:
            logging.warning(f"Failed to identify new features for expert {expert.name}, chunk {chunk_idx}: {e}")
            new_features = []
            feature_values = {}
        
        return ExpertChunkNewFeatures(
            expert_name=expert.name,
            chunk_index=chunk_idx,
            new_features=new_features,
            new_feature_values=feature_values
        )

    def _consolidate_features(
        self, 
        chunk_features: List[Dict[str, str]], 
        paper_title: str
    ) -> Dict[str, str]:
        """Consolidate extracted features across all chunks and experts"""
        consolidated = {}
        
        for feature_name in self.board.sota_table.features:
            # Collect all values for this feature
            values = []
            for chunk_feature_dict in chunk_features:
                value = chunk_feature_dict.get(feature_name, "")
                if value and value != "Not Available":
                    values.append(value)
            
            if values:
                # Consolidate using LLM
                prompt = build_feature_consolidation_prompt(
                    feature_name, paper_title, values
                )
                consolidated_model = self.json_generator.generate_json(prompt, StringResponseModel)
                consolidated_value = consolidated_model.response
            else:
                consolidated_value = "Not Available"
            
            consolidated[feature_name] = consolidated_value
        
        return consolidated

    def _consolidate_new_features(
        self, 
        chunk_new_features: List[ExpertChunkNewFeatures], 
        paper_title: str
    ) -> Dict[str, str]:
        """Consolidate new features identified across all chunks and experts"""
        # Collect all new features and their values
        feature_values_map = {}
        
        for chunk_new_feature in chunk_new_features:
            for feature_name in chunk_new_feature.new_features:
                if feature_name not in feature_values_map:
                    feature_values_map[feature_name] = []
                
                value = chunk_new_feature.new_feature_values.get(feature_name, "")
                if value and value != "Not Available":
                    feature_values_map[feature_name].append(value)
        
        # Consolidate each new feature
        consolidated = {}
        for feature_name, values in feature_values_map.items():
            if values:
                prompt = build_feature_consolidation_prompt(
                    feature_name, paper_title, values
                )
                consolidated_model = self.json_generator.generate_json(prompt, StringResponseModel)
                consolidated_value = consolidated_model.response
            else:
                consolidated_value = "Not Available"
            
            consolidated[feature_name] = consolidated_value
        
        return consolidated

    def _extract_year_from_id(self, doc_id: str) -> int:
        """Extract year from document ID"""
        try:
            if doc_id and '-' in doc_id:
                year_str = doc_id.split('-')[0]
                if year_str.isdigit() and 1900 < int(year_str) < 2100:
                    return int(year_str)
            return 2023
        except (ValueError, IndexError, AttributeError):
            return 2023

    def _extract_domain(self, doc: Document, experts: List[Expert]) -> str:
        """Extract the domain/field of the paper"""
        prompt = build_domain_extraction_prompt(
            doc.title, doc.authors, doc.abstract
        )
        
        try:
            domain_model = self.json_generator.generate_json(prompt, StringResponseModel)
            return domain_model.response
        except Exception as e:
            logging.error(f"Failed to extract domain: {e}")
            return ""

    def _display_thoughts(self, thoughts: List[str]) -> str:
        """Format thesis thoughts for display"""
        return "\n".join([f"- {thought}" for thought in thoughts])
    
    def _extract_new_feature_values_from_chunk(
        self,
        chunk,
        chunk_idx: int,
        new_features: List[str],
        doc: Document
    ) -> Dict[str, str]:
        """Extract values for specific new features from a document chunk"""
        # Create a simple prompt to extract values for these specific features
        prompt = build_new_feature_identification_prompt(
            "Expert in extracting features from academic papers", # Generic description
            self.board.thesis_knowledge.description,
            doc.title,
            doc.authors,
            chunk.chunk,
            new_features  # Only looking for the new features
        )
        
        try:
            response_model = self.json_generator.generate_json(prompt, DictResponseModel)
            response = response_model.data
            feature_values = response.get("feature_values", {})
            
            # Only keep values for the specific new features we're looking for
            filtered_values = {f: feature_values.get(f, "") for f in new_features if f in feature_values}
            return filtered_values
            
        except Exception as e:
            logging.warning(f"Failed to extract new feature values for chunk {chunk_idx} of {doc.title}: {e}")
            return {}