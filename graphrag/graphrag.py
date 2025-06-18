import random
from typing import List, Tuple, Any, Dict
from pydantic import BaseModel, Field

from tqdm import tqdm

import networkx as nx
from networkx.algorithms.community import louvain_communities

import re
from collections import defaultdict
import concurrent.futures

from entities.document import Document
from graphrag.interfaces.json_generator import JsonGenerator
from graphrag.interfaces.text_embedder import TextEmbedder
from graphrag.knowledge_graph import KnowledgeGraph
from graphrag.prompts.extract_graph import initial_extract_graph_prompt
from graphrag.prompts.extract_claims import extract_claims_prompt
from graphrag.prompts.summary_descriptions import summary_descriptions_prompt
from graphrag.prompts.summary_community import summary_community_prompt
from graphrag.models.entity_relationship import EntityRelationshipModel
from graphrag.models.summary_community import SummaryCommunityModel
from graphrag.models.text_unit import TextUnit
from graphrag.models.claim_list import ClaimListModel
from graphrag.utils.text_chunking import chunk_document
from graphrag.models.graph_types import Entity, Relationship, Claim, EntityType, Community, CommunityReport
from graphrag.models.summary_description import SummaryDescriptionModel

# Import the moved models
from graphrag.models.initial_answer_model import InitialAnswerModel
from graphrag.models.follow_up_questions_model import FollowUpQuestionsModel
from graphrag.models.local_search_model import LocalSearchModel
from graphrag.models.final_response_model import FinalResponseModel


class GraphRag:
    """
    Builds a Graph-RAG from a collection of documents, following the GraphRAG Knowledge Model workflow.
    """
    def __init__(self, text_embedder: TextEmbedder, json_generator: JsonGenerator, small_json_generator: JsonGenerator = None,max_tokens: int = 3000, overlap_tokens: int = 50, low_consume: bool = True, use_rag: bool = True):
        """
        Initializes the GraphRAGBuilder with the necessary components.
        """
        self.text_embedder = text_embedder
        self.json_generator = json_generator
        self.small_json_generator = small_json_generator
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.low_consume = low_consume
        self.use_rag = use_rag

    def build_knowledge_graph(self, documents: List[Document]) -> KnowledgeGraph:
        kg = KnowledgeGraph(documents=documents)

        #==============================================================================================================================
        # Phase 1: Compose TextUnits using threads
        def process_document(doc):
            text_units: List[TextUnit] = self._chunk_document(doc, max_tokens=self.max_tokens, overlap_tokens=self.overlap_tokens)
            return text_units
            
        all_text_units = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_doc = {executor.submit(process_document, doc): doc for doc in documents}
            for future in tqdm(concurrent.futures.as_completed(future_to_doc), total=len(documents), desc="Processing documents"):
                text_units = future.result()
                all_text_units.extend(text_units)
                
        # Add all text units to the knowledge graph
        for tu in all_text_units:
            kg.add_text_unit(tu)
        #==============================================================================================================================      
        # Phase 2: Graph Extraction (Entities, Relationships, Covariates)
        all_entities: List[Entity] = []
        all_relationships: List[Relationship] = []
        textunit_entities: Dict[str, List[Entity]] = {}

        if self.low_consume:
            tu_union: TextUnit = None
            for idx, tu in enumerate(tqdm(kg.text_units, desc="Extracting entities/relationships"), 1):
                if tu_union is None:
                    tu_union = tu

                if tu_union.number_tokens + tu.number_tokens < self.max_tokens-100 and idx != len(kg.text_units):
                    tu_union.text += "\n"*3 + "#"*30 + "\n"*3 + tu.text
                    tu_union.number_tokens += tu.number_tokens+50
                else:
                    entities, relationships = self.extract_entities_and_relationships_from_textunit(tu_union)
                    all_entities.extend(entities)
                    all_relationships.extend(relationships)
                    # Save entities for this textunit_id (if tu_union has id)
                    
                    textunit_entities[tu_union.unit_id] = entities
                    tu_union = None
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future_to_tu = {
                    executor.submit(self.extract_entities_and_relationships_from_textunit, tu): tu
                    for tu in kg.text_units
                }
                for idx, future in enumerate(tqdm(concurrent.futures.as_completed(future_to_tu), total=len(kg.text_units), desc="Extracting entities/relationships (multi-threaded)"), 1):
                    tu = future_to_tu[future]
                    entities, relationships = future.result()
                    all_entities.extend(entities)
                    all_relationships.extend(relationships)
                    textunit_entities[tu.unit_id] = entities
        print("Finished extracting entities/relationships.")
        merged_entities: Dict[str, Tuple[EntityType,List[str]]] = {}
        entity_type_map: Dict[EntityType, List[str]] = {}
        for ent in all_entities:
            key = ent.name
            if key not in merged_entities:
                merged_entities[key] = (ent.type, [ent.description])
            else:
                merged_entities[key][1].append(ent.description)
            if ent.type not in entity_type_map:
                entity_type_map[ent.type] = [ent.description]
            else:
                entity_type_map[ent.type].append(ent.description)
        # entities_for_claims: List[Entity] = [Entity(name=name, type=type_, description="") for name, (type_, _) in merged_entities.items()]
        # for tu in kg.text_units:
        #     covariates = self.extract_covariates_from_textunit(tu, entities_for_claims)
        #     for cov in covariates:
        #         kg.add_covariate(cov)

        summarized_entities: List[Entity] = []
        for name, (type_, descriptions) in merged_entities.items():
            summary: str = self.summary_descriptions(descriptions)
            entity = Entity(name=name, type=type_, description=summary)
            summarized_entities.append(entity)

        summarized_entities_types: Dict[EntityType, str] = {}
        for type_, descriptions in entity_type_map.items():
            summary = self.summary_descriptions(descriptions)
            summarized_entities_types[type_] = summary

        for entity in summarized_entities:
            kg.add_entity(entity)

        for textunit_id, entities in textunit_entities.items():
            kg.add_textunits_entities(textunit_id, entities)

        merged_relationships: Dict[Tuple[str, str], List[str]] = {}
        for rel in all_relationships:
            key = (rel.source, rel.target)
            if key not in merged_relationships:
                merged_relationships[key] = [rel.description]
            else:
                merged_relationships[key].append(rel.description)
        summarized_relationships: List[Relationship] = []
        for (source, target), descriptions in merged_relationships.items():
            summary = self.summary_descriptions(descriptions)
            summarized_relationships.append(Relationship(source=source, target=target, description=summary))
        for rel in summarized_relationships:
            kg.add_relationship(rel)
        #==============================================================================================================================
        # Phase 3: Graph Augmentation (Community Detection)
        communities = self.detect_communities(kg)
        for comm in communities:
            kg.add_community(comm)
        #==============================================================================================================================
        # # Phase 4: Community Summarization
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_comm = {
                executor.submit(self.summarize_community, comm, kg): comm
                for comm in kg.communities
            }
            for future in concurrent.futures.as_completed(future_to_comm):
                comm = future_to_comm[future]
                report = future.result()
                comm.report = report
                kg.add_community_report(report)
        #==============================================================================================================================

        return kg

    def update_knowledge_graph(self, kg: KnowledgeGraph, docs: List[Document]):
        """
        Incrementally update the knowledge graph with new documents.
        """
        # 1. Chunk new documents and add text units using threads
        def process_document(doc):
            return self._chunk_document(doc, max_tokens=self.max_tokens, overlap_tokens=self.overlap_tokens)
            
        new_text_units = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_doc = {executor.submit(process_document, doc): doc for doc in docs}
            for future in tqdm(concurrent.futures.as_completed(future_to_doc), total=len(docs), desc="Processing new documents"):
                tus = future.result()
                for tu in tus:
                    kg.add_text_unit(tu)
                    new_text_units.append(tu)

        # 2. Extract entities and relationships from new text units
        all_entities = []
        all_relationships = []
        textunit_entities = {}
        if self.low_consume:
            tu_union = None
            for idx, tu in enumerate(new_text_units, 1):
                if tu_union is None:
                    tu_union = tu
                if tu_union.number_tokens + tu.number_tokens < self.max_tokens-100 and idx != len(new_text_units):
                    tu_union.text += "\n"*3 + "#"*30 + "\n"*3 + tu.text
                    tu_union.number_tokens += tu.number_tokens+50
                else:
                    entities, relationships = self.extract_entities_and_relationships_from_textunit(tu_union)
                    all_entities.extend(entities)
                    all_relationships.extend(relationships)
                    textunit_entities[tu_union.unit_id] = entities
                    tu_union = None
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future_to_tu = {
                    executor.submit(self.extract_entities_and_relationships_from_textunit, tu): tu
                    for tu in new_text_units
                }
                for idx, future in enumerate(tqdm(concurrent.futures.as_completed(future_to_tu), total=len(new_text_units), desc="Extracting entities/relationships (multi-threaded)"), 1):
                    tu = future_to_tu[future]
                    entities, relationships = future.result()
                    all_entities.extend(entities)
                    all_relationships.extend(relationships)
                    textunit_entities[tu.unit_id] = entities

        # 3. Merge new entities and relationships
        merged_entities = {e.name: e for e in kg.entities}
        for ent in all_entities:
            if ent.name in merged_entities:
                # Optionally update description (e.g., merge summaries)
                merged_entities[ent.name].description += f"; {ent.description}"
            else:
                merged_entities[ent.name] = ent
        # Summarize entity descriptions
        for name, ent in merged_entities.items():
            descs = ent.description.split(';')
            ent.description = self.summary_descriptions(descs)
        kg.entities = list(merged_entities.values())

        # Update textunit-entity mapping
        for textunit_id, entities in textunit_entities.items():
            kg.add_textunits_entities(textunit_id, entities)

        # Merge relationships
        rel_key = lambda r: (r.source, r.target)
        merged_relationships = {(r.source, r.target): r for r in kg.relationships}
        for rel in all_relationships:
            key = rel_key(rel)
            if key in merged_relationships:
                merged_relationships[key].description += f"; {rel.description}"
            else:
                merged_relationships[key] = rel
        # Summarize relationship descriptions
        for rel in merged_relationships.values():
            descs = rel.description.split(';')
            rel.description = self.summary_descriptions(descs)
        kg.relationships = list(merged_relationships.values())

        # 4. Re-run community detection and summarization
        kg.communities = []
        kg.community_reports = []
        communities = self.detect_communities(kg)
        for comm in communities:
            kg.add_community(comm)
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_comm = {
                executor.submit(self.summarize_community, comm, kg): comm
                for comm in kg.communities
            }
            for future in concurrent.futures.as_completed(future_to_comm):
                comm = future_to_comm[future]
                report = future.result()
                comm.report = report
                kg.add_community_report(report)

        # Optionally, update covariates if needed (not shown here)

    def _chunk_document(self, doc: Document, max_tokens=100000, overlap_tokens=50) -> List[TextUnit]:
        """
        Chunk document's content into semantically meaningful chunks using spaCy, with overlap,
        and convert them to TextUnit objects.

        Args:
            doc (Document): Input document to be chunked.
            max_tokens (int): Maximum number of tokens per chunk.
            overlap_tokens (int): Number of tokens to overlap between chunks.
        Returns:
            List[TextUnit]: list of TextUnit objects created from the chunks.
        """
        return chunk_document(self.text_embedder,doc, max_tokens=max_tokens, overlap_tokens=overlap_tokens)

    def extract_entities_and_relationships_from_textunit(self, text_unit: TextUnit, example: str = "") -> Tuple[List[Entity], List[Relationship]]:
        """
        Extract entities and relationships from a text unit using the LLM and merge results as per the GraphRAG workflow.

        Args:
            text_unit (TextUnit): The text unit object to process.
            example (str, optional): Example text to guide the extraction. Defaults to "".

        Returns:
            Tuple[List[Entity], List[Relationship]]: Lists of merged entities and relationships.
        """
        entity_types = ",".join([e.value for e in EntityType])
        prompt = initial_extract_graph_prompt(text_unit.text, entity_types, example)
        entity_relationships:EntityRelationshipModel = self.json_generator.generate_json(prompt, EntityRelationshipModel)

        return entity_relationships.entities, entity_relationships.relationships

    def extract_covariates_from_textunit(self, text_unit: TextUnit,entities: List[Entity]) -> List[Claim]:
        """
        Extract claims (covariates) from a text unit using the LLM and the Claim model.
        
        Args:
            text_unit (TextUnit): The text unit object from which to extract claims.
            
        Returns:
            List[Claim]: List of claims extracted from the text unit.
        """
        prompt = extract_claims_prompt(text_unit.text,entities)
        if self.json_generator is not None:
            claim_list = self.json_generator.generate_json(prompt, ClaimListModel)
            return claim_list.claims
        else:
            return []

    def _kg_to_networkx(self, kg: KnowledgeGraph):
        """
        Convert the KnowledgeGraph to a networkx.Graph for community detection.
        Nodes are entity names, edges are relationships.
        """
        G = nx.Graph()
        node_names = [e.name for e in kg.entities]
        G.add_nodes_from(node_names)
        edges = [(rel.source, rel.target) for rel in kg.relationships if rel.source in node_names and rel.target in node_names]
        G.add_edges_from(edges)
        return G

    def detect_communities(self, kg: KnowledgeGraph, min_community_size: int = 3) -> List[Community]:
        """
        Detect hierarchical communities using the Louvain algorithm from networkx (networkx.algorithms.community.louvain_communities).
        Returns a flat list of Community objects (one per detected community).
        """
        G = self._kg_to_networkx(kg)
        communities = []
        entity_type_lookup = {e.name: e.type for e in kg.entities}
        def recursive_louvain(graph,parent_id=None, level=0):
            comms = louvain_communities(graph, resolution = 1+0.2*level, seed=42)
            i = 0
            for idx, members in enumerate(comms):
                if len(members) < min_community_size:
                    continue
                i = i + 1
                this_comm_id = f"L{level}_C{i}"
                member_tuples = [(m, entity_type_lookup.get(m, EntityType.CONCEPT)) for m in members]
                community = Community(
                    id=this_comm_id,
                    level=level,
                    members=member_tuples,
                    parent=parent_id,
                    report=None
                )
                communities.append(community)
                if len(members) > min_community_size*3:
                    subgraph = graph.subgraph(members)
                    if subgraph.number_of_nodes() > min_community_size and level<5:
                        recursive_louvain(subgraph, parent_id=this_comm_id, level=level+1)
        recursive_louvain(G)
        return communities

    def summarize_community(self, community: Community, kg: KnowledgeGraph) -> CommunityReport:
        """
        Generate a report for a community using the LLM, referencing key entities and relationships.
        Returns a CommunityReport object.
        """
        members = set([m[0] for m in community.members])
        key_entities = [e for e in kg.entities if e.name in members]
        key_relationships = [r for r in kg.relationships if r.source in members and r.target in members]

        prompt = summary_community_prompt(key_entities, key_relationships)
        if self.small_json_generator is not None:
            response: SummaryCommunityModel = self.small_json_generator.generate_json(prompt, SummaryCommunityModel)
            summary = response.summary
            key_entities = response.key_entities
            key_relationships = response.key_relationships
        else:
            entity_descs = "; ".join([e.description for e in key_entities])
            rel_descs = "; ".join([r.description for r in key_relationships])
            summary = f"Entities: {entity_descs}. Relationships: {rel_descs}"
            summary = summary[:5000] + ("..." if len(summary) > 2000 else "")

        return CommunityReport(summary=summary, key_entities=key_entities, key_relationships=key_relationships)

    def summary_descriptions(self, descriptions: List[str]) -> str:
        """
        Summarize a list of descriptions using the LLM (JsonGenerator) and a Pydantic model.

        Args:
            descriptions (List[str]): List of description strings to summarize.

        Returns:
            str: A concise summary generated by the LLM.
        """
        if len(descriptions) == 0:
            return descriptions[0]

        prompt = summary_descriptions_prompt(descriptions)
        if self.small_json_generator is not None:
            response: SummaryDescriptionModel = self.small_json_generator.generate_json(prompt, SummaryDescriptionModel)
            return response.summary
        else:
            random.shuffle(descriptions)
            summary = "; ".join(descriptions)
            return summary[:5000] + ("..." if len(summary) > 5000 else "")
        
    
    
    def find_documents(self, query: str, kg: KnowledgeGraph, k: int) -> List[Document]:
        """
        Find documents relevant to a query using the knowledge graph.
        Uses cosine similarity between the response embedding and each text unit embedding,
        then aggregates per document and sorts by average similarity.
        """
        response = query
        response_embedding = self.text_embedder.embed(response)

        # Map: doc_id -> [similarities]
        doc_similarities = defaultdict(list)
        doc_id_to_doc = {}
        for tu in kg.text_units:
            tu_embedding = tu.embedding
            # Cosine similarity
            dot = sum(a * b for a, b in zip(response_embedding.vector, tu_embedding.vector))
            norm1 = sum(a * a for a in response_embedding.vector) ** 0.5
            norm2 = sum(b * b for b in tu_embedding.vector) ** 0.5
            similarity = dot / (norm1 * norm2 + 1e-8)
            doc_similarities[tu.document_id].append(similarity)
        for doc in kg.documents:
            doc_id_to_doc[doc.id] = doc
        
        doc_avg_sim = []
        for doc_id, sims in doc_similarities.items():
            avg_sim = sum((sim)**3 for sim in sims) / len(sims)
            doc_avg_sim.append((doc_id, avg_sim))
        # Sort by similarity descending
        doc_avg_sim.sort(key=lambda x: x[1], reverse=True)
        # Return top k documents
        top_docs = [doc_id_to_doc[doc_id] for doc_id, _ in doc_avg_sim[:k] if doc_id in doc_id_to_doc]
        return top_docs


    def filter_relevant_text_units(self, text_units, query, top_n=3):
        response_embedding = self.text_embedder.embed(query)

        # Map: doc_id -> [similarities]
        tu_similarities = {}

        for tu in text_units:
            tu_embedding = tu.embedding
            # Cosine similarity
            dot = sum(a * b for a, b in zip(response_embedding.vector, tu_embedding.vector))
            norm1 = sum(a * a for a in response_embedding.vector) ** 0.5
            norm2 = sum(b * b for b in tu_embedding.vector) ** 0.5
            similarity = dot / (norm1 * norm2 + 1e-8)
            tu_similarities[tu] = similarity

        # Sort by similarity descending
        sorted_tus = sorted(tu_similarities.items(), key=lambda x: x[1], reverse=True)
        # Return top n text units
        top_tus = [tu for tu, _ in sorted_tus[:top_n]]
        return top_tus

    def respond(self, query: str, kg: KnowledgeGraph, c: int = 3) -> str:
        """
        Improved DRIFT search: All reasoning steps use LLM prompts and JsonGenerator.
        """
        drift_intro = (
            "Combining Local and Global Search\n\n"
            "GraphRAG uses LLMs to create knowledge graphs and summaries from unstructured text, "
            "enabling both global overviews and detailed local exploration. DRIFT search (Dynamic Reasoning and Inference with Flexible Traversal) "
            "combines global and local search to generate comprehensive answers.\n\n"
            "---\n"
        )

        # Phase A: Global Search
        relevant_communities = self._find_relevant_communities(query, kg, c)
        community_reports = [comm.report for comm in relevant_communities if comm.report]
        community_summaries = [r.summary for r in community_reports]
        community_entities = [e for r in community_reports for e in r.key_entities]
        community_relationships = [rel for r in community_reports for rel in r.key_relationships]

        # Build a global prompt
        global_prompt = (
            f"User Query: {query}\n"
            f"Community Summaries:\n"
            + "\n".join(f"- {s}" for s in community_summaries[:5])
            + "\nKey Entities:\n"
            + ", ".join(f"{e.name} ({e.type.value})" for e in community_entities[:10])
            + "\nKey Relationships:\n"
            + "\n".join(f"{rel.source} -> {rel.target}: {rel.description}" for rel in community_relationships[:10])
            + "\n\n"
            "Based on the above, provide:\n"
            "- A comprehensive answer to the query\n"
            "- 3-5 key insights\n"
            "- A confidence score (0.0-1.0)\n"
            "- Reasoning steps\n"
        )
        initial_answer_model = self.json_generator.generate_json(global_prompt, InitialAnswerModel)
        initial_answer = initial_answer_model.answer
        confidence_score = initial_answer_model.confidence_score

        # Phase A2: Generate follow-up questions using LLM
        followup_prompt = (
            f"Given the answer:\n{initial_answer}\n"
            "Generate 3-5 follow-up questions that would help refine or deepen the answer. "
            "For each, specify the type (entity, relationship, temporal, causal) and a priority score (0.0-1.0)."
        )
        followup_model = self.json_generator.generate_json(followup_prompt, FollowUpQuestionsModel)
        follow_up_questions = followup_model.questions

        # Phase B: Local Search for each follow-up
        intermediate_responses = []
        for i, follow_up_q in enumerate(follow_up_questions):
            # Build a local search prompt

            relevant_text_units = self.filter_relevant_text_units(kg.text_units, follow_up_q)

            local_prompt = (
                f"User Follow-up Question: {follow_up_q}\n"
                f"Relevant Text Units: {[tu.text[:100] for tu in relevant_text_units]}\n"
                "Provide:\n"
                "- A detailed answer\n"
                "- List of evidence sources\n"
                "- Confidence score (0.0-1.0)\n"
                "- Key entities mentioned"
            )
            local_model = self.json_generator.generate_json(local_prompt, LocalSearchModel)
            local_answer = local_model.answer
            local_confidence = local_model.confidence_score
            intermediate_responses.append({
                'question': follow_up_q,
                'answer': local_answer,
                'confidence': local_confidence
            })
            if local_confidence < 0.3:
                break

        # Phase C: Output Hierarchy and Summary (LLM-driven)
        summary_prompt = (
            f"Original Query: {query}\n"
            f"Global Answer: {initial_answer}\n"
            f"Local Refinements:\n"
            + "\n".join(f"Q: {r['question']}\nA: {r['answer']}\nConfidence: {r['confidence']:.2f}" for r in intermediate_responses)
            + "\n\n"
            "Summarize the findings, assess overall confidence, and provide recommendations for further exploration."
        )
        final_model = self.json_generator.generate_json(summary_prompt, FinalResponseModel)

        # Compose the final response
        response_parts = []
        response_parts.append("=" * 80)
        response_parts.append("DRIFT SEARCH RESPONSE")
        response_parts.append(f"Query: {query}")
        response_parts.append("=" * 80)
        response_parts.append("\n🌍 **GLOBAL OVERVIEW**")
        response_parts.append(f"Confidence: {confidence_score:.1%}")
        response_parts.append("-" * 50)
        response_parts.append(initial_answer)
        if intermediate_responses:
            response_parts.append(f"\n🎯 **LOCAL REFINEMENTS** ({len(intermediate_responses)} Follow-up Explorations)")
            response_parts.append("-" * 50)
            sorted_responses = sorted(intermediate_responses, key=lambda x: x['confidence'], reverse=True)
            for i, resp in enumerate(sorted_responses, 1):
                confidence_icon = "🟢" if resp['confidence'] > 0.6 else "🟡" if resp['confidence'] > 0.3 else "🔴"
                response_parts.append(f"\n{i}. {confidence_icon} **Q:** {resp['question']}")
                response_parts.append(f"   **Confidence:** {resp['confidence']:.1%}")
                response_parts.append(f"   **A:** {resp['answer']}")
        response_parts.append("\n📊 **SUMMARY**")
        response_parts.append("-" * 20)
        response_parts.append(final_model.executive_summary)
        response_parts.append(final_model.global_insights)
        response_parts.append("\n".join(final_model.local_findings))
        response_parts.append(final_model.confidence_assessment)
        response_parts.append("\n".join(final_model.recommendations))
        response_parts.append("\n" + "=" * 80)
        
        response = []

        response.append(final_model.executive_summary)
        response.append(final_model.global_insights)
        response.append("\n".join(final_model.local_findings))
        response.append(final_model.confidence_assessment)
        response.append("\n".join(final_model.recommendations))

        return "\n".join(response)

    def _find_relevant_communities(self, query: str, kg: KnowledgeGraph, k: int) -> List[Community]:
        """Find the top K most semantically relevant communities for the query."""
        
        if not kg.communities or not kg.community_reports:
            return []
        
        query_lower = query.lower()
        community_scores = []
        
        for community in kg.communities:
            if not community.report:
                continue
                
            score = 0.0
            
            # Score based on community report summary
            if community.report.summary:
                summary_lower = community.report.summary.lower()
                # Simple keyword matching (in real implementation, use semantic similarity)
                common_words = set(query_lower.split()) & set(summary_lower.split())
                score += len(common_words) * 2
            
            # Score based on key entities
            for entity in community.report.key_entities:
                if entity.name.lower() in query_lower:
                    score += 3
                if any(word in entity.description.lower() for word in query_lower.split()):
                    score += 1
            
            # Score based on key relationships
            for rel in community.report.key_relationships:
                if rel.source.lower() in query_lower or rel.target.lower() in query_lower:
                    score += 2
                if any(word in rel.description.lower() for word in query_lower.split()):
                    score += 1
            
            community_scores.append((community, score))
        
        # Sort by score and return top K
        community_scores.sort(key=lambda x: x[1], reverse=True)
        return [comm for comm, score in community_scores[:k]]

    def _generate_initial_answer(self, query: str, communities: List[Community]) -> Tuple[str, float]:
        """Generate initial broad answer from community reports."""
        
        if not communities:
            return f"I don't have sufficient information to answer: {query}", 0.1
        
        # Aggregate information from community reports
        key_points = []
        entities_mentioned = set()
        relationships_found = []
        
        for community in communities:
            if not community.report:
                continue
                
            # Extract key information from community summary
            if community.report.summary:
                key_points.append(community.report.summary)
            
            # Collect relevant entities
            for entity in community.report.key_entities:
                entities_mentioned.add(f"{entity.name} ({entity.type.value})")
            
            # Collect relevant relationships
            relationships_found.extend([
                f"{rel.source} → {rel.target}: {rel.description}"
                for rel in community.report.key_relationships
            ])
        
        # Build initial answer
        answer_parts = []
        answer_parts.append(f"Based on the available information, here's what I found regarding '{query}':")
        
        if key_points:
            answer_parts.append("\n**Community Insights:**")
            for i, point in enumerate(key_points[:3], 1):  # Limit to top 3
                answer_parts.append(f"{i}. {point}")
        
        if entities_mentioned:
            answer_parts.append(f"\n**Key Entities:** {', '.join(list(entities_mentioned)[:5])}")
        
        if relationships_found:
            answer_parts.append(f"\n**Key Relationships:**")
            for rel in relationships_found[:3]:  # Limit to top 3
                answer_parts.append(f"• {rel}")
        
        # Calculate confidence based on information richness
        confidence = min(0.9, len(key_points) * 0.2 + len(entities_mentioned) * 0.1 + len(relationships_found) * 0.1)
        
        return "\n".join(answer_parts), confidence

    def _generate_follow_up_questions(self, original_query: str, initial_answer: str, communities: List[Community]) -> List[str]:
        """Generate follow-up questions to guide local search."""
        
        follow_ups = []
        
        # Extract entities and concepts from communities for targeted questions
        entities = set()
        concepts = set()
        
        for community in communities:
            if not community.report:
                continue
                
            for entity in community.report.key_entities:
                if entity.type.value in ['PERSON', 'ORGANIZATION', 'LOCATION']:
                    entities.add(entity.name)
                elif entity.type.value in ['CONCEPT', 'EVENT']:
                    concepts.add(entity.name)
        
        # Generate different types of follow-up questions
        query_words = original_query.lower().split()
        
        # Entity-focused questions
        for entity in list(entities)[:2]:
            follow_ups.append(f"What specific role does {entity} play in relation to {original_query}?")
        
        # Relationship questions
        if len(entities) >= 2:
            entity_list = list(entities)
            follow_ups.append(f"How are {entity_list[0]} and {entity_list[1]} connected?")
        
        # Concept deepening questions
        for concept in list(concepts)[:2]:
            follow_ups.append(f"Can you provide more details about {concept} in the context of {original_query}?")
        
        # Temporal questions
        if any(word in query_words for word in ['when', 'timeline', 'history', 'development']):
            follow_ups.append(f"What is the timeline or sequence of events related to {original_query}?")
        
        # Causal questions
        if any(word in query_words for word in ['why', 'cause', 'reason', 'impact']):
            follow_ups.append(f"What are the underlying causes or impacts related to {original_query}?")

        return follow_ups[:3]  # Limit to 3 follow-up questions

    def _local_search(self, query: str, kg: KnowledgeGraph, context_communities: List[Community]) -> Tuple[str, float]:
        """Perform local search to find specific information."""
        
        query_lower = query.lower()
        relevant_info = []
        confidence_factors = []
        
        # Search through text units for direct mentions
        for text_unit in kg.text_units:
            if any(word in text_unit.text.lower() for word in query_lower.split()):
                relevant_info.append(f"From document {text_unit.document_id}: {text_unit.text[:200]}...")
                confidence_factors.append(0.3)
        
        # Search through entities for detailed descriptions
        for entity in kg.entities:
            if (entity.name.lower() in query_lower or 
                any(word in entity.description.lower() for word in query_lower.split())):
                relevant_info.append(f"**{entity.name}** ({entity.type.value}): {entity.description}")
                confidence_factors.append(0.4)
        
        # Search through relationships for connections
        for rel in kg.relationships:
            if (any(word in rel.description.lower() for word in query_lower.split()) or
                rel.source.lower() in query_lower or rel.target.lower() in query_lower):
                relevant_info.append(f"**Relationship**: {rel.source} → {rel.target}: {rel.description}")
                confidence_factors.append(0.3)
        
        # Search through claims/covariates
        for claim in kg.covariates:
            if (claim.subject.lower() in query_lower or claim.object.lower() in query_lower or
                any(word in claim.claim_description.lower() for word in query_lower.split())):
                relevant_info.append(f"**Claim**: {claim.subject} - {claim.claim_description}")
                confidence_factors.append(0.2)
        
        # Build local answer
        if not relevant_info:
            return f"No specific local information found for: {query}", 0.1
        
        answer = f"**Local Search Results for:** {query}\n\n"
        answer += "\n\n".join(relevant_info[:10])

        # Calculate confidence
        confidence = min(0.8, sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.1)
        
        return answer, confidence

    def _build_hierarchical_response(self, original_query: str, initial_answer: str, 
                                    initial_confidence: float, intermediate_responses: List[Dict]) -> str:
        """Build the final hierarchical response structure."""
        
        response_parts = []
        
        # Header
        response_parts.append("=" * 80)
        response_parts.append(f"DRIFT SEARCH RESPONSE")
        response_parts.append(f"Query: {original_query}")
        response_parts.append("=" * 80)
        
        # Phase A: Global Overview
        response_parts.append("\n🌍 **GLOBAL OVERVIEW** (Community-Level Insights)")
        response_parts.append(f"Confidence: {initial_confidence:.1%}")
        response_parts.append("-" * 50)
        response_parts.append(initial_answer)
        
        # Phase B: Local Refinements
        if intermediate_responses:
            response_parts.append(f"\n🎯 **LOCAL REFINEMENTS** ({len(intermediate_responses)} Follow-up Explorations)")
            response_parts.append("-" * 50)
            
            # Sort by confidence (highest first)
            sorted_responses = sorted(intermediate_responses, key=lambda x: x['confidence'], reverse=True)
            
            for i, resp in enumerate(sorted_responses, 1):
                confidence_icon = "🟢" if resp['confidence'] > 0.6 else "🟡" if resp['confidence'] > 0.3 else "🔴"
                response_parts.append(f"\n{i}. {confidence_icon} **Q:** {resp['question']}")
                response_parts.append(f"   **Confidence:** {resp['confidence']:.1%}")
                response_parts.append(f"   **A:** {resp['answer']}")
        
        # Summary and Recommendations
        response_parts.append(f"\n📊 **SUMMARY**")
        response_parts.append("-" * 20)
        
        avg_confidence = (initial_confidence + sum(r['confidence'] for r in intermediate_responses)) / (len(intermediate_responses) + 1)
        response_parts.append(f"Overall Confidence: {avg_confidence:.1%}")
        response_parts.append(f"Information Sources: {len(intermediate_responses) + 1} layers of analysis")
        
        if avg_confidence < 0.5:
            response_parts.append("⚠️  **Note:** Limited information available. Consider refining your query or checking if more relevant documents are needed.")
        
        response_parts.append("\n" + "=" * 80)
        
        return "\n".join(response_parts)
