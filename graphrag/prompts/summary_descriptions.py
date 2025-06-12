from typing import List
from graphrag.models.summary_community import SummaryCommunityModel

def summary_descriptions_prompt(descriptions: List[str]) -> str:
    return (
        "Summarize the following descriptions into a concise, informative sentence or two. "
        "Return the result as JSON with a 'summary' field.\n"
        f"JSON schema: {SummaryCommunityModel.model_json_schema()}\n"
        "Descriptions:\n" + "\n".join(descriptions)

    )
