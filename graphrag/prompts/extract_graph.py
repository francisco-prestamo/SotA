from graphrag.models.entity_relationship import EntityRelationshipModel

def initial_extract_graph_prompt(text: str, entity_types: str, examples: str = "") -> str:
    return f"""
    -Goal-
    Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

    -Steps-
    1. Identify all entities. For each identified entity, extract the following information:
    - entity_name: Name of the entity, capitalized
    - entity_type: One of the following types: [{entity_types}]
    - entity_description: Comprehensive description of the entity's attributes and activities

    2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
    For each pair of related entities, extract the following information:
    - source_entity: name of the source entity, as identified in step 1
    - target_entity: name of the target entity, as identified in step 1
    - relationship_description: explanation as to why you think the source entity and the target entity are related to each other
    - relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity

    3. Return output as a JSON object matching this JSON schema:
    {EntityRelationshipModel.model_json_schema()}

    ######################
    -Examples-
    ######################
    {examples}

    ######################
    -Real Data-
    ######################
    Text: {text}
    ######################
    Output JSON:
    """