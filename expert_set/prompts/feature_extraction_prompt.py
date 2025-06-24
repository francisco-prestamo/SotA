from typing import List

def build_feature_extraction_prompt(
    expert_description: str,
    paper_description: str,
    document_title: str,
    document_authors: List[str],
    chunk_content: str,
    features_to_extract: List[str]
) -> str:
    """Build prompt for extracting specific features from a document chunk"""
    features_list = "\n".join([f"- {feature}" for feature in features_to_extract])
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

This is a description of your target research paper

### DESCRIPTION ###
{paper_description}
### END ###

As such you're specialized in a specific domain which relates with the target research paper, this is your description:
### DESCRIPTION ###
{expert_description}
### END ###

You are reading the documents chunk by chunk, this is one of the chunks for a given research paper.

Your task is to extract specific features from the given document chunk. You must provide a value for each feature listed below, i.e. a short description of how it was explored, implemented, demonstrated, etc. in the document,
whatever is appropriate given the description of the feature

### FEATURES TO EXTRACT ###
{features_list}
### END ###

### DOCUMENT INFORMATION ###
- Title: {document_title}
- Authors: {authors_str}
### END ###

### DOCUMENT CHUNK ###
{chunk_content}
### END ###

INSTRUCTIONS:
1. For each feature listed above, extract the most relevant information from the chunk
2. If a feature is not explicitly mentioned or cannot be inferred, respond with "Not Available"
3. Provide concise but informative values
4. Focus on factual information from the text
"""
