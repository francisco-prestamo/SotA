from typing import List


def build_feature_extraction_prompt(
        expert_name: str,
        expert_description: str,
        document_title: str,
        document_authors: List[str],
        chunk_content: str,
        features_to_extract: List[str]
) -> str:
    """Build prompt for extracting specific features from a document chunk, with examples and clear instructions to provide a brief description/value for each feature."""
    features_list = "\n".join([f"- {feature}" for feature in features_to_extract])
    authors_str = ", ".join(document_authors)

    # Add concrete, simpler examples for AutoML, LLM, and other fields
    example_features = '''Example features and values:
# Machine Learning / AI
- AutoML for LLM: "Yes" or "No"
- LLMs for AutoML: "Yes" or "No"
- Tool: "AutoGraph", "AutoGen", "GPT-NAS", etc.
- Pre-training: "Yes" or "No"
- Fine-tuning: "Yes" or "No"
- In-Context Learning: "Yes" or "No"
- Model Selection: "Yes" or "No"
- HPO: "Grid Search", "Bayesian Opt.", "Random Search", etc.
- Meta-learning: "Yes" or "No"
- Multi-modal: "Yes" or "No"
- Application: e.g., "Medical Image Segmentation", "Text Classification"
- Dataset: e.g., "ImageNet", "CIFAR-10"
- Evaluation Metric: e.g., "Accuracy", "F1-score"
- Baseline: e.g., "ResNet-50", "EfficientNet-B0"

# Biology
- Gene Target: e.g., "BRCA1", "TP53"
- Organism: e.g., "Drosophila", "Mouse"
- Experimental Method: e.g., "Western Blot", "PCR"
- Pathway: e.g., "MAPK/ERK", "PI3K/AKT"

# Physics
- Experimental Setup: e.g., "Double-slit", "Laser Interferometer"
- Theoretical Model: e.g., "Standard Model", "Quantum Harmonic Oscillator"
- Measurement: e.g., "Spin", "Energy"

# Chemistry
- Compound: e.g., "Aspirin", "Benzene"
- Reaction Type: e.g., "SN2", "Esterification"
- Catalyst: e.g., "Pd/C", "H2SO4"

# Medicine
- Patient Population: e.g., "Adults", "Children"
- Intervention: e.g., "Metformin", "Placebo"
- Outcome: e.g., "HbA1c Reduction", "Survival Rate"

# Social Sciences
- Survey Instrument: e.g., "Likert Scale", "Questionnaire"
- Sample Size: e.g., "500", "1000"
- Statistical Test: e.g., "ANOVA", "t-test"
'''

    return f"""You are {expert_name}, {expert_description}.

Your task is to extract specific features from the given document chunk. For each feature, provide a simple value as implemented or discussed in the paper.

FEATURES TO EXTRACT:
{features_list}

DOCUMENT INFORMATION:
- Title: {document_title}
- Authors: {authors_str}

DOCUMENT CHUNK:
{chunk_content}

INSTRUCTIONS:
1. For each feature listed above, extract the most relevant information from the chunk.
2. For each feature, provide a simple value (e.g., Yes/No, or select the main option or name).
3. If a feature is not explicitly mentioned or cannot be inferred, respond with "Not Available".
4. Use the following format:
{{
  "feature_name": "value",
  ...
}}

{example_features}

Respond as a JSON object with feature names as keys and extracted values as the response."""