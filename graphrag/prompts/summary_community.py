from graphrag.models.summary_community import SummaryCommunityModel

def summary_community_prompt(entities, relationships):
    return (
        "Generate an executive summary for the following community of entities and relationships. "
        "Describe the main topics, connections, and any notable patterns. "
        "Return the result as JSON with a 'summary' field.\n"
        f"JSON schema: {SummaryCommunityModel.model_json_schema()}\n"
        f"Entities: {entities}\n"
        f"Relationships: {relationships}\n"
    )
