from graphrag.models.entity_relationship import EntityRelationshipModel


def get_graph_extraction_examples() -> str:
    return """
Example 1:
Text: "Dr. Sarah Chen, a renowned AI researcher at Stanford University, published a groundbreaking paper on neural networks. She collaborated with Microsoft's research team led by Dr. James Wilson to develop new machine learning algorithms."

Output:
{
  "entities": [
    {
      "name": "Dr. Sarah Chen",
      "type": "PERSON",
      "description": "Renowned AI researcher specializing in neural networks and machine learning"
    },
    {
      "name": "Stanford University",
      "type": "ORGANIZATION", 
      "description": "Academic institution where Dr. Sarah Chen conducts research"
    },
    {
      "name": "Microsoft",
      "type": "ORGANIZATION",
      "description": "Technology company with research team working on machine learning algorithms"
    },
    {
      "name": "Dr. James Wilson",
      "type": "PERSON",
      "description": "Research team leader at Microsoft focusing on machine learning algorithms"
    },
    {
      "name": "Neural Networks",
      "type": "CONCEPT",
      "description": "AI technology that Dr. Sarah Chen published research about"
    }
  ],
  "relationships": [
    {
      "source": "Dr. Sarah Chen",
      "target": "Stanford University",
      "description": "employed as researcher"
    },
    {
      "source": "Dr. Sarah Chen",
      "target": "Neural Networks",
      "description": "published research on"
    },
    {
      "source": "Dr. Sarah Chen",
      "target": "Dr. James Wilson",
      "description": "collaborated on research"
    },
    {
      "source": "Dr. James Wilson",
      "target": "Microsoft",
      "description": "leads research team"
    },
    {
      "source": "Microsoft",
      "target": "Stanford University",
      "description": "research collaboration"
    }
  ]
}

Example 2:
Text: "The climate summit in Paris brought together world leaders to discuss environmental policies. President Johnson announced new carbon reduction targets, while the European Union pledged $50 billion for renewable energy initiatives."

Output:
{
  "entities": [
    {
      "name": "Climate Summit",
      "type": "EVENT",
      "description": "International conference focused on environmental policies and climate action"
    },
    {
      "name": "Paris",
      "type": "LOCATION",
      "description": "City where the climate summit took place"
    },
    {
      "name": "President Johnson",
      "type": "PERSON",
      "description": "Political leader who announced carbon reduction targets at the summit"
    },
    {
      "name": "European Union",
      "type": "ORGANIZATION",
      "description": "Political and economic union that pledged funding for renewable energy"
    },
    {
      "name": "Carbon Reduction Targets",
      "type": "CONCEPT",
      "description": "Environmental goals announced by President Johnson"
    },
    {
      "name": "Renewable Energy",
      "type": "CONCEPT",
      "description": "Clean energy technology receiving EU investment"
    }
  ],
  "relationships": [
    {
      "source": "Climate Summit",
      "target": "Paris",
      "description": "held in location"
    },
    {
      "source": "President Johnson",
      "target": "Climate Summit",
      "description": "participated in event"
    },
    {
      "source": "European Union",
      "target": "Climate Summit", 
      "description": "participated in event"
    },
    {
      "source": "President Johnson",
      "target": "Carbon Reduction Targets",
      "description": "announced policy"
    },
    {
      "source": "European Union",
      "target": "Renewable Energy",
      "description": "pledged funding for"
    }
  ]
}
"""


def initial_extract_graph_prompt(text: str, entity_types: str, examples: str = "") -> str:
    example_text = get_graph_extraction_examples()

    # Handle entity_types properly to avoid empty brackets
    if entity_types and entity_types.strip():
        entity_types_instruction = f"One of the following types: {entity_types}"
    else:
        entity_types_instruction = "PERSON, ORGANIZATION, LOCATION, EVENT, or CONCEPT"

    return f"""
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
   - name: Name of the entity, capitalized
   - type: {entity_types_instruction}
   - description: Comprehensive description of the entity's attributes and activities

2. From the entities identified in step 1, identify all pairs of (source, target) that are *clearly related* to each other.
   For each pair of related entities, extract the following information:
   - source: name of the source entity, as identified in step 1
   - target: name of the target entity, as identified in step 1
   - description: brief explanation of the relationship between source and target entities

3. Format your response as valid JSON with "entities" and "relationships" arrays.

######################
-Examples-
######################
{example_text}

######################
-Real Data-
######################
Text: {text}

Output the result as valid JSON only. Ensure the JSON is complete and properly closed with all brackets and braces.
"""