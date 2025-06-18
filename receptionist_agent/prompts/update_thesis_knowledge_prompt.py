def update_thesis_knowledge_prompt(current_knowledge, user_query, response, update_model):
    """
    Prompt for updating thesis knowledge based on user query and response.
    
    Args:
        current_knowledge: The current knowledge about the thesis topic
        user_query: The user's query
        response: The response provided to the query
        update_model: The model class for the updated knowledge
        
    Returns:
        str: The formatted prompt
    """
    prompt = f"""As an AI thesis advisor, your task is to update our knowledge about a thesis topic based on a new interaction with the user.

Current Knowledge:
Description: {current_knowledge.description}

Current Knowledge Points:
"""
    
    for thought in current_knowledge.thoughts:
        prompt += f"- {thought}\n"
    
    prompt += f"""
New User Query:
"{user_query}"

Response Provided:
"{response}"

Based on this interaction, please extract new knowledge about the thesis topic and update the existing knowledge.
If the user's query and the response provide new information about the thesis topic, add it to the knowledge.
If the description needs to be refined, please update it.

Output the updated knowledge following the provided model schema. Include both existing knowledge points that are still relevant and new knowledge points extracted from this interaction.
"""
    
    return prompt
