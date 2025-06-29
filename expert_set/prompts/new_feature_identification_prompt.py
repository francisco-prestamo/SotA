from typing import List

def build_new_feature_identification_prompt(
    paper_description: str,
    expert_description: str,
    document_title: str,
    document_authors: List[str],
    chunk_content: str,
    existing_features: List[str],
) -> str:
    """Build prompt for identifying NEW FEATURE NAMES ONLY in a document chunk."""
    existing_features_list = "\n".join(
        [f"- {feature}" for feature in existing_features]
    )
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
### DESCRIPTION ###
{expert_description}
### END ###

You are reading the documents chunk by chunk, this is one of the chunks for a given research paper.

Your task is to identify NEW features in the given document chunk that are NOT already covered by the existing features.

### EXISTING FEATURES (DO NOT include these) ###
{existing_features_list}
### END ###

### DOCUMENT INFORMATION ###
- Title: {document_title}
- Authors: {authors_str}
### END ####

### DOCUMENT CHUNK ###
{chunk_content}
### END ###

INSTRUCTIONS:
1. Look for important characteristics, metrics, methods, or properties mentioned in the chunk
2. Only identify features that are NOT already covered by the existing features list
3. Focus on features that would be valuable for comparing papers in a SOTA table
4. Provide both the feature name and its value from this chunk
5. Limit to the 3 most important new features

These are some example `feature name -> value` pairs, representing structured metadata typically extracted from research papers for building a state-of-the-art comparison table in the field of computer science, and specifically AI-related
studies

### EXAMPLES ###
Example 1:

- AutoML for LLM → No
- LLMs for AutoML → No
- AutoML → No
- Pre-training → No
- Fine-tuning → No
- Inferences → No
- Meta-learning → Yes
- Tool → None
- LLM → Yes
- Model Selection → No
- HPO → No
- Prompt Tuning → No
- In-Context Learning → No
- Multi-modal → Yes
- Datasets → "Amazon-531", "DBPedia-298"

Example 2:

- AutoML for LLM → Yes
- LLMs for AutoML → Yes
- AutoML → Yes
- Pre-training → No
- Fine-tuning → No
- Inferences → No
- Meta-learning → No
- Tool → AutoGen
- LLM → Yes
- Model Selection → No
- HPO → No
- Prompt Tuning → No
- In-Context Learning → No
- Multi-modal → No
- Datasets → Not specified
### END ###

As you can see they are concise, short and direct, note that for different fields of research, the names and
values could change a bit, perhaps more than 'Yes' and 'No' would be necessary to explain each feature for a 
given paper, but the conciseness should remain.

**An empty list of new features is a VALID RESPONSE, as a matter of fact it is PREFERRED, you should ONLY 
identify features that are both CLEARLY MISSING from the currently existing features, CLEARLY RELEVANT to the 
description of your target research paper and CLEARLY APPEARING in the document chunk.**

This is a description of your target research paper, maintain extracted features relevant to it, output no features
if you find no clearly relevant ones
### DESCRIPTION ###
{paper_description}
### END ###

Respond as a JSON object with:
- "reasoning": a string that could show your reasoning of why you believe that features appearing in the chunk are 
CLEARLY RELEVANT to your target research paper, and CLEARLY MISSING from the already existing features, 
conversely, it could show your reasoning as to why the current chunk has no features that are clearly relevant to
the target research paper and are not redundant with the already existing ones.
- "new_features": array of new feature names"""


def build_feature_value_extraction_prompt(
    feature_name: str,
    document_title: str,
    document_authors: List[str],
    chunk_content: str,
) -> str:
    """Build prompt for extracting the value of a specific feature from a document chunk."""
    authors_str = ", ".join(document_authors)

    return f"""
You are an expert, sequentially reading a paper, trying to assert whether or not a feature is present in each
of the chunks of text you iterate through


These are some example `feature name -> value` pairs, representing structured metadata typically extracted from research papers for building a state-of-the-art comparison table in the field of computer science, and specifically AI-related
studies

### EXAMPLES ###
Example 1:

- AutoML for LLM → No
- LLMs for AutoML → No
- AutoML → No
- Pre-training → No
- Fine-tuning → No
- Inferences → No
- Meta-learning → Yes
- Tool → None
- LLM → Yes
- Model Selection → No
- HPO → No
- Prompt Tuning → No
- In-Context Learning → No
- Multi-modal → Yes
- Datasets → "Amazon-531", "DBPedia-298"

Example 2:

- AutoML for LLM → Yes
- LLMs for AutoML → Yes
- AutoML → Yes
- Pre-training → No
- Fine-tuning → No
- Inferences → No
- Meta-learning → No
- Tool → AutoGen
- LLM → Yes
- Model Selection → No
- HPO → No
- Prompt Tuning → No
- In-Context Learning → No
- Multi-modal → No
- Datasets → Not specified
### END ###

As you can see they are concise, short and direct, note that for different fields of research, the names and
values could change a bit, perhaps more than 'Yes' and 'No' would be necessary to explain each feature for a 
given paper, but the conciseness should remain.

Assert the value for the feature '{feature_name}' from the given document chunk.

DOCUMENT INFORMATION:
- Title: {document_title}
- Authors: {authors_str}

FEATURE TO EXTRACT: {feature_name}

DOCUMENT CHUNK:
{chunk_content}

INSTRUCTIONS:
1. Look for information related to '{feature_name}' in the chunk.
2. Provide a concise, specific value (e.g., "Yes"/"No", specific tool name, metric value, etc.).
3. If the feature is not mentioned or unclear, respond with "Not Available".
4. Keep the response brief and factual.

Examples of good responses:
- For "Tool Used": "GPT-4" or "ResNet-50" or "PyTorch"
- For "Pre-training Applied": "Yes" or "No"
- For "Dataset Used": "ImageNet" or "CIFAR-10"
- For "Evaluation Metric": "Accuracy" or "F1-score"

Respond with just the value as a simple string.
"""

