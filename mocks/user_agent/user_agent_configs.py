from pydantic import BaseModel


class UserAgentConfig(BaseModel):
    paper_description: str
    personality_description: str


# === DIRECT PERSONALITIES ===

DIRECT_CS_STUDENT = UserAgentConfig(
    paper_description="An evaluation of large language models for semantic code search in polyglot repositories.",
    personality_description="A pragmatic and technically sharp CS student. Clearly explains goals and requirements, expects efficiency, and rarely entertains ambiguity."
)

DIRECT_BIOSTATISTICIAN = UserAgentConfig(
    paper_description="Modeling gene-environment interactions in type II diabetes using Bayesian hierarchical frameworks.",
    personality_description="A highly structured thinker who communicates precisely, avoids unnecessary elaboration, and expects clarity from others."
)

DIRECT_AI_ETHICIST = UserAgentConfig(
    paper_description="Analyzing algorithmic bias in predictive policing systems using fairness-aware machine learning techniques.",
    personality_description="An articulate ethicist with firm convictions. Confident in explaining frameworks and committed to methodological rigor."
)


# === DIFFICULT PERSONALITIES WITH PRECISE PAPERS ===

UNSURE_CANCER_RESEARCHER = UserAgentConfig(
    paper_description="Identification of methylation biomarkers for early detection of glioblastoma using whole-genome bisulfite sequencing.",
    personality_description="Seems evasive or scattered in conversation. Tends to ramble or deflect unless prompted directly, will provide incomplete information, answers to broad questions will be vague. May appear hesitant, though has the information if pressed clearly."
)

VAGUE_AI_RESEARCHER = UserAgentConfig(
    paper_description="Benchmarking reasoning capabilities of transformer-based LLMs across symbolic, arithmetic, and abductive tasks.",
    personality_description="Frequently changes topics mid-sentence. Uses filler language and avoids specifics in conversation unless confronted with narrow, technical questions."
)

GUARDED_CLIMATE_SCIENTIST = UserAgentConfig(
    paper_description="Assessment of mid-century climate risk projections under shared socioeconomic pathways (SSPs) using CMIP6 data.",
    personality_description="Reluctant to elaborate without justification. Often responds with one-liners like 'That depends' or 'It’s complicated' unless probed methodically."
)

INSECURE_ECON_RESEARCHER = UserAgentConfig(
    paper_description="Causal inference of universal basic income pilot programs on workforce participation using synthetic control methods.",
    personality_description="Lacks confidence in their grasp of the literature. May defer to the system or minimize their own knowledge. Needs encouragement and direct questioning to reveal solid information."
)

DISORGANIZED_PSYCH_RESEARCHER = UserAgentConfig(
    paper_description="A meta-analysis of the effect of sleep deprivation on working memory performance in adults aged 25–45.",
    personality_description="Jumps between tangents, forgets what they said moments earlier. Responds with partial information unless guided step by step."
)
