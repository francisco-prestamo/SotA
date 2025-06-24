import json
from typing import List, Dict
import logging
from pydantic import create_model
import concurrent.futures
from threading import Lock

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
from .models import NewFeaturesListModel

from .prompts.feature_extraction_prompt import build_feature_extraction_prompt
from .prompts.new_feature_identification_prompt import build_new_feature_identification_prompt, \
    build_feature_value_extraction_prompt
from .prompts.feature_consolidation_prompt import build_feature_consolidation_prompt
from .prompts.domain_extraction_prompt import build_domain_extraction_prompt
from .prompts.addition_summary_prompt import build_addition_summary_prompt
from .prompts.search_query_synthesis_prompt import build_search_query_synthesis_prompt

from typing import List


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
        new_docs = self.recoverer_agent.recover_docs(summary_query, self.k)

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
        print(f"Extracting features from {doc.title}...")
        extraction_result = self._extract_paper_features(doc, experts)
         # Update SOTA table features if new ones were found
        new_features_added = []
        for new_feature in extraction_result.new_features:
            if new_feature not in self.board.sota_table.features:
                self.board.sota_table.features.append(new_feature)
                new_features_added.append(new_feature)
        
        # If we added new features, process all existing documents to check for these features
        if new_features_added:
            pass

        all_features = {}
        
        # Add existing features
        for feature in self.board.sota_table.features:
            if feature in extraction_result.old_features:
                all_features[feature] = {"value": extraction_result.old_features[feature]}
            elif feature in extraction_result.new_features:
                all_features[feature] = {"value": extraction_result.new_features[feature]}
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
        print(f"Paper {doc.title}")
        print("*"*100)
        # Add to SOTA table
        self.board.sota_table.document_features.append((doc, paper_features))

    def _extract_paper_features(self, doc: Document, experts: List[Expert]) -> PaperFeatureExtraction:
        """Extract features from a paper using multiple experts across chunks"""
        from expert_set.utils.document_chunking import chunk_document
        chunks = chunk_document(doc,window_size=500)
        
        chunk_features = []
        chunk_new_features = []
        has_existing_features = bool(self.board.sota_table.features)
        print(f"Processing {len(chunks)} chunks with multithreading")
        
        # Create thread-safe data structures with locks
        chunk_features_lock = Lock()
        chunk_new_features_lock = Lock()
        
        # Function to process a single chunk with an expert
        def process_chunk(chunk_info):
            chunk_idx, chunk = chunk_info
            results = []
            
            # We're using only the first expert for now as in the original code
            expert = experts[0]
                
            # Extract existing features only if any exist in the SOTA table
            if has_existing_features:
                existing_features = self._extract_features_from_chunk(
                    expert, doc, chunk, chunk_idx
                )
                print(f"Chunk {chunk_idx} - Existing features:")
                print(existing_features)
                with chunk_features_lock:
                    chunk_features.append(existing_features)
                    
            # Identify new features
            new_features = self._identify_new_features_from_chunk(
                expert, doc, chunk, chunk_idx
            )
            print(f"Chunk {chunk_idx} - New features:")
            print(new_features)
            with chunk_new_features_lock:
                chunk_new_features.append(new_features)
                
            return True
        
        # Execute processing in parallel with thread pool
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Create a list of (chunk_idx, chunk) tuples
            chunk_items = list(enumerate(chunks))
            # Submit all chunks for processing and wait for completion
            futures = [executor.submit(process_chunk, chunk_info) for chunk_info in chunk_items]
            concurrent.futures.wait(futures)


        all_old_features = self._consolidate_features(chunk_features, doc.title) if chunk_features else {}
        
        # Consolidate all new features from different chunks
        all_new_features = self._consolidate_new_features(chunk_new_features, doc.title)

        print("_"*120)
        print("Paper:")

        print(json.dumps({"old_features": all_old_features, "new_features": all_new_features}, indent=4, default=str))
        return PaperFeatureExtraction(
            document=doc,
            old_features=all_old_features,
            new_features=all_new_features
        )

    def _extract_features_from_chunk(
        self, 
        expert: Expert, 
        doc: Document, 
        chunk, 
        chunk_idx: int
    ) -> Dict[str, str]:
        """Extract existing SOTA table features from a chunk using an expert, expecting a brief description/value for each feature."""
        prompt = build_feature_extraction_prompt(
            expert.expert_model.description,
            self.board.thesis_knowledge.description,
            doc.title,
            doc.authors,
            chunk.chunk,
            self.board.sota_table.features
        )
        FeaturesModel = create_features_extraction_model(self.board.sota_table.features)
        try:
            response = self.json_generator.generate_json(prompt, FeaturesModel)
            extracted_features = {}
            for feature in self.board.sota_table.features:
                # Accept string or dict, but prefer string (brief description)
                value = getattr(response, feature, "")
                if isinstance(value, dict) and 'value' in value:
                    value = value['value']
                extracted_features[feature] = value
        except Exception as e:
            logging.warning(f"Failed to extract features for expert {expert.name}, chunk {chunk_idx}: {e}")
            extracted_features = {feature: "Not Available" for feature in self.board.sota_table.features}
        return extracted_features

    def _identify_new_features_from_chunk(
        self,
        expert: Expert,
        doc: Document,
        chunk,
        chunk_idx: int
    ) -> ExpertChunkNewFeatures:
        """Identify new features in a chunk using an expert"""
        prompt_names = build_new_feature_identification_prompt(
            expert.expert_model.description,
            self.board.thesis_knowledge.description,
            doc.title,
            doc.authors,
            chunk.chunk,
            self.board.sota_table.features
        )
        
        print(f"Extracting new feature names from paper chunk {chunk_idx}:")
        try:
            response_names = self.json_generator.generate_json(prompt_names, NewFeaturesListModel)
            new_features = response_names.new_features
            print(f"Found new features: {new_features}")
        except Exception as e:
            logging.warning(f"Failed to identify new feature names for expert {expert.name}, chunk {chunk_idx}: {e}")
            new_features = []

        # Step 2: For each new feature, extract its value
        feature_values = {}
        if new_features:
            print(f"Extracting values for {len(new_features)} new features...")
            for feature in new_features:
                value_prompt = build_feature_value_extraction_prompt(
                    feature,
                    doc.title,
                    doc.authors,
                    chunk.chunk
                )
                try:
                    value_response = self.json_generator.generate_json(value_prompt, StringResponseModel)
                    feature_values[feature] = value_response.response
                    print(f"  {feature}: {value_response.response}")
                except Exception as e:
                    logging.warning(f"Failed to extract value for new feature '{feature}' for expert {expert.name}, chunk {chunk_idx}: {e}")
                    feature_values[feature] = "Not Available"
        
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
        
        # Step 1: Collect all candidate features from all chunks
        all_candidate_features = []
        for chunk_new_feature in chunk_new_features:
            all_candidate_features.extend(chunk_new_feature.new_features)
        
        # Step 2: Use JSON generator to consolidate and deduplicate feature names
        consolidation_prompt = f"""
        Paper: {paper_title}
        
        The following features were identified from different chunks of the paper:
        {chr(10).join([f"- {feature}" for feature in all_candidate_features])}
        
        Existing features in the SOTA table:
        {chr(10).join([f"- {feature}" for feature in self.board.sota_table.features])}
        
        Please consolidate these candidate features by:
        1. Merging features that refer to the same concept (e.g., "model accuracy" and "accuracy" should be merged)
        2. Removing features that are already covered by existing SOTA table features
        3. Providing a final list of unique, new features that you see relevants at least one or two and no more than 7
        
        Return only the consolidated list of new feature names that don't already exist in the SOTA table.
        """
        
        try:
            consolidated_features_response = self.json_generator.generate_json(
                consolidation_prompt, 
                NewFeaturesListModel
            )
            global_new_features = consolidated_features_response.new_features
        except Exception as e:
            logging.warning(f"Failed to consolidate feature names for paper {paper_title}: {e}")
            # Fallback to simple deduplication
            existing_features = set(self.board.sota_table.features)
            global_new_features = list(set(all_candidate_features) - existing_features)
        
        print(f"Global new features (not in SOTA table): {global_new_features}")
        
        # If no new features, return empty dict
        if not global_new_features:
            return {}
        
        # Step 3: Create dynamic model for these global new features
        GlobalNewFeaturesModel = create_features_extraction_model(global_new_features)
        
        # Step 4: Collect all values for each global new feature across chunks
        feature_values_map = {}
        for feature_name in global_new_features:
            feature_values_map[feature_name] = []
            
        # Populate values from all chunks
        for chunk_new_feature in chunk_new_features:
            for feature_name in global_new_features:
                if feature_name in chunk_new_feature.new_features:
                    value = chunk_new_feature.new_feature_values.get(feature_name, "")
                    if value and value != "Not Available":
                        feature_values_map[feature_name].append(value)
        
        # Step 5: Create consolidated prompt with all values for all features
        all_feature_values_text = []
        for feature_name, values in feature_values_map.items():
            if values:
                values_text = f"{feature_name}: {'; '.join(values)}"
            else:
                values_text = f"{feature_name}: Not Available"
            all_feature_values_text.append(values_text)
        
        # Build consolidation prompt for all features at once
        prompt = f"""
        Paper: {paper_title}
        
        The following features and their values were extracted from different chunks of the paper:
        {chr(10).join(all_feature_values_text)}
        
        Please consolidate these feature values into final values for each feature. 
        For each feature, provide a single consolidated value that best represents 
        the information found across all chunks.
        
        If multiple values exist for a feature, synthesize them into one coherent description.
        If no values exist for a feature, respond with "Not Available".
        """
        
        try:
            # Step 6: Use the dynamic model to get consolidated values
            consolidated_response = self.json_generator.generate_json(prompt, GlobalNewFeaturesModel)
            
            # Extract values from the response
            consolidated = {}
            for feature_name in global_new_features:
                value = getattr(consolidated_response, feature_name, "Not Available")
                # Handle both string and dict responses
                if isinstance(value, dict) and 'value' in value:
                    value = value['value']
                consolidated[feature_name] = value
                
        except Exception as e:
            logging.warning(f"Failed to consolidate new features for paper {paper_title}: {e}")
            # Fallback: use first available value for each feature
            consolidated = {}
            for feature_name in global_new_features:
                values = feature_values_map.get(feature_name, [])
                consolidated[feature_name] = values[0] if values else "Not Available"
        
        print(f"Consolidated new feature values: {json.dumps(consolidated, indent=2)}")
        
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
