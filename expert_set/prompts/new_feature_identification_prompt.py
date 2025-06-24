import json
from typing import List
from ..models import ExpertChunkNewFeatures

from board.board import ThesisKnowledgeModel

def build_new_feature_identification_prompt(
    expert_description: str,
    paper_description: str,
    document_title: str,
    document_authors: List[str],
    chunk_content: str,
    existing_features: List[str]
) -> str:
    """Build prompt for identifying new features in a document chunk"""
    existing_features_list = "\n".join([f"- {feature}" for feature in existing_features])
    authors_str = ", ".join(document_authors)
    
    return f"""
You are an expert, part of a group who is tasked with examining a set of research papers and extracting features
from it, in order to compose a state of the art section for a specific target research paper, the state of the art 
section is the section of a research paper that explores similar investigations on the subject by other 
researchers, and compares them with their own work. The features should relate to the research paper, exemplifying
the ways in which the same subject was addressed (i.e. if the paper you're reading solves a problem stated in the
paper you're composing the state of the art section for in a different way, or shows a result that is similar
to the ones shown in your target research paper). The objective is to create a state of the art TABLE, which
has as rows the documents, and as columns, features, with a short description of why they were deemed detected
in a given research paper

As such you're specialized in a specific domain which relates with the target research paper, this is your description:
### YOUR DESCRIPTION ###
{expert_description}
### END ###

Here is a description of the target research paper:
### TARGET RESEARCH PAPER DESCRIPTION ###
{paper_description}
### END ###

You are reading the documents chunk by chunk, this is one of the chunks for a given research paper.
Your task is to identify NEW features in the given document chunk that are NOT already covered by the existing features.
You must provide a value for each of the new features you add, i.e. a short description of how it was explored, 
implemented, demonstrated, etc. in the document, whatever is appropriate given the description of the feature

### DOCUMENT INFORMATION (of the current chunk) ###
- Title: {document_title}
- Authors: {authors_str}
### END ###

### EXISTING FEATURES (DO NOT include these) ###
{existing_features_list}
### END ###

### DOCUMENT CHUNK ###
{chunk_content}
### END ###

INSTRUCTIONS:
1. Look for important characteristics, metrics, methods, or properties mentioned in the chunk
2. Only identify features that are NOT already covered by the existing features list
3. Focus on features that would be valuable for comparing papers in a state of the art section
4. Provide both the feature name and its value from this chunk
5. Limit to the 3 most important new features

**An empty list is a VALID response, as a matter of fact, it is PREFERRED, ONLY output new
features if they're CLEARLY RELEVANT to your target research paper and are CLEARLY MISSING
from the existing ones**

Respond as a JSON object with:
- "new_features": array of new feature names
- "feature_values": object mapping each new feature name to its value in this chunk
As per the following schema:
{json.dumps(ExpertChunkNewFeatures.model_json_schema(), indent=2)}
"""


