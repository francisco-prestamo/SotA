def scrapper_selection_prompt(query: str, scrapper_infos: list, result_schema) -> str:
    descriptions = "\n".join([f"{info['name']}: {info['description']}" for info in scrapper_infos])
    return f"""
Given the following user query and a list of available document scrappers with their descriptions, for each scrapper, indicate if it should be used for searching (true or false). Return a JSON object with a dictionary where the key is the scrapper name and the value is true or false, and a brief reasoning.

Query: {query}

Available scrappers:
{descriptions}

Return your answer as a JSON object matching this schema:
{result_schema.model_json_schema()}
"""
