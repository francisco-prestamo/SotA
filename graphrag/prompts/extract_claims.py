from typing import List
from graphrag.models.claim_list import ClaimListModel
from graphrag.models.graph_types import Entity

def extract_claims_prompt(text: str, entities: List[Entity]) -> str:
    """
    Returns a prompt for extracting claims (covariates) from a text unit, following the detailed user requirements.
    """
    json_schema_str = f"JSON schema: {ClaimListModel.model_json_schema()}"
    text_str = f"Text: {text}"
    entities_str = f"{('; '.join([e.name for e in entities]))}" if entities else "Entities: None"
    return (
        f"""
        -Target activity-
        You are an intelligent assistant that helps a human analyst to analyze claims against certain entities presented in a text document.

        -Goal-
        Given a text document that is potentially relevant to this activity, an entity specification, and a claim description, extract all entities that match the entity specification and all claims against those entities.

        -Steps-
        1. Extract all named entities that match the predefined entity specification. Entity specification can either be a list of entity names or a list of entity types.
        2. For each entity identified in step 1, extract all claims associated with the entity. Claims need to match the specified claim description, and the entity should be the subject of the claim.

        For each claim, extract the following information:
        - Subject: name of the entity that is subject of the claim, capitalized. The subject entity is one that committed the action described in the claim. Subject needs to be one of the named entities identified in step 1.
        - Object: name of the entity that is object of the claim, capitalized. The object entity is one that either reports/handles or is affected by the action described in the claim. If object entity is unknown, use **NONE**.
        - Claim Type: overall category of the claim, capitalized. Name it in a way that can be repeated across multiple text inputs, so that similar claims share the same claim type
        - Claim Status: **TRUE**, **FALSE**, or **SUSPECTED**. TRUE means the claim is confirmed, FALSE means the claim is found to be False, SUSPECTED means the claim is not verified.
        - Claim Description: Detailed description explaining the reasoning behind the claim, together with all the related evidence and references.
        - Claim Date: Period (start_date, end_date) when the claim was made. Both start_date and end_date should be in ISO-8601 format. If the claim was made on a single date rather than a date range, set the same date for both start_date and end_date. If date is unknown, return **NONE**.
        - Claim Source Text: List of **all** quotes from the original text that are relevant to the claim.
        Return the result as JSON with a 'claims' field, which is a list of Claim objects.
        {json_schema_str}

        -Entities Names-
        {entities_str}

        -Text-
        {text_str}
        """
    )
