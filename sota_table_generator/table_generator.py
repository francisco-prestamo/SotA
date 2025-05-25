import pandas as pd
from typing import List, Dict, Any
from entities.document import Document
from sota_table_generator.interfaces.json_generator import JsonGenerator
from pydantic import BaseModel

class ExpertDescriptionModel(BaseModel):
    name: str
    description: str

class ExpertListModel(BaseModel):
    experts: list[ExpertDescriptionModel]

class PaperFeaturesModel(BaseModel):
    authors: List[str]
    title: str
    year: int
    domain: str
    features: List[str]

class ExpertAgent:
    """
    Base class for an expert agent that processes a set of documents and extracts specific information.
    """
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

class SotAMultiAgentTableGenerator:
    """
    Multi-agent SotA table generator. Each expert agent extracts different aspects from the documents.
    """
    def __init__(self, json_generator: JsonGenerator):
        self.json_generator = json_generator
        self.experts = []

    def extract_paper_features(self, document: Document) -> PaperFeaturesModel:
        """
        For each expert agent, use the LLM to extract PaperFeaturesModel for each document, then merge results with a general LLM call.
        """
        expert_features = []
        for expert in self.experts:
            prompt = (
                f"You are an expert in {expert.description}.\n"
                f"Given the following document, extract the following fields as JSON: {PaperFeaturesModel.schema_json(indent=2)}.\n"
                f"Document: {document.content}\n"
            )
            features = self.json_generator.generate_json(prompt, PaperFeaturesModel)
            expert_features.append(features)

        merge_prompt = (
            "Given the following extracted features from multiple experts for the same paper, merge them into a single, most complete PaperFeaturesModel as JSON.\n"
            f"Expert Features: {[f.dict() for f in expert_features]}\n"
        )
        final_features = self.json_generator.generate_json(merge_prompt, PaperFeaturesModel)
        return final_features

    def generate_table(self, documents: List[Document]) -> pd.DataFrame:
        """
        Dynamically generates expert agents based on the documents using a JSON generator,
        then each expert processes the documents and their results are merged into a DataFrame.
        """
        prompt = (
            "Given a set of documents, generate a list of expert agent descriptions.\n"
            "Each expert should focus on a different field related to the docs.\n"
            "Return a JSON object with a list of experts, each with a 'name' and 'description'.\n"
            f"Documents Abstracts: {[doc.abstract for doc in documents]}\n"
        )
        expert_list = self.json_generator.generate_json(prompt, ExpertListModel)
        
        self.experts = []

        for expert_desc in expert_list.experts:
            self.experts.append(ExpertAgent(expert_desc.name, expert_desc.description))

        rows = []
        for doc in documents:
            features = self.extract_paper_features(doc)
            rows.append(features.dict())
        df = pd.DataFrame(rows)

        return df



