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

# Import the moved models
from .models.paper_addition_result_model import PaperAdditionResult
from .models.expert_chunk_new_features_model import ExpertChunkNewFeatures
from .models.paper_feature_extraction_model import PaperFeatureExtraction

# Import the moved prompts
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
        expert_names = [expert.name for expert in experts]
        expert_search_reasoning_model = build_expert_search_reasoning_model(expert_names)
        
        # Build context for reasoning
        sota_md = sota_table_to_markdown(self.board.sota_table)
        thesis_desc = self.board.thesis_knowledge.description
        thesis_thoughts = self._display_thoughts(self.board.thesis_knowledge.thoughts)
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
        summary_query = self.json_generator.generate_json(synthesis_prompt, str)
        
        # Recover new documents
        new_docs = self.recoverer_agent.recover_docs(summary_query, self.board.knowledge_graph, self.k)
        
        if not new_docs:
            logging.warning("No new documents were recovered.")
            return PaperAdditionResult(papers_added=[], summary="No new documents were recovered.")
        
        # Add each document to the SOTA table
        added_titles = []
        original_features = set(self.board.sota_table.features)
        
        for doc in new_docs:
            self._add_paper_to_sota_table(doc, experts)
            added_titles.append(doc.title)
        
        # Identify new features that were added
        new_features = [f for f in self.board.sota_table.features if f not in original_features]
        
        # Generate summary
        summary_prompt = build_addition_summary_prompt(
            added_titles, expert_names, new_features
        )
        summary = self.json_generator.generate_json(summary_prompt, str)
        
        return PaperAdditionResult(papers_added=added_titles, summary=summary)

    def _add_paper_to_sota_table(self, doc: Document, experts: List[Expert]) -> None:
        """Add a single paper to the SOTA table by extracting features"""
        # Extract features from the document
        extraction_result = self._extract_paper_features(doc, experts)
        
        # Update SOTA table features if new ones were found
        for new_feature in extraction_result.consolidated_new_features:
            if new_feature not in self.board.sota_table.features:
                self.board.sota_table.features.append(new_feature)
                # Add empty values for existing papers
                for existing_doc, existing_features in self.board.sota_table.document_features:
                    if new_feature not in existing_features.features:
                        existing_features.features[new_feature] = {"value": "Not Available"}
        
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
            expert.name,
            expert.expert_model.description,
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
            expert.name,
            expert.expert_model.description,
            doc.title,
            doc.authors,
            chunk.chunk,
            self.board.sota_table.features
        )
        
        try:
            response = self.json_generator.generate_json(prompt, dict)
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
                consolidated_value = self.json_generator.generate_json(prompt, str)
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
                consolidated_value = self.json_generator.generate_json(prompt, str)
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
            domain = self.json_generator.generate_json(prompt, str)
            return domain
        except Exception as e:
            logging.error(f"Failed to extract domain: {e}")
            return ""

    def _display_thoughts(self, thoughts: List[str]) -> str:
        """Format thesis thoughts for display"""
        return "\n".join([f"- {thought}" for thought in thoughts])