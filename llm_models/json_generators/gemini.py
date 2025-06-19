from google import genai
from graphrag.interfaces.json_generator import JsonGenerator as GraphRagJsonGen, T
from board.board import JsonGenerator as BoardJsonGen
from expert_set.interfaces import JsonGenerator as ExpertSetJsonGen
from recoverer_agent.interfaces import JsonGenerator as RecovJsonGen
from receptionist_agent.interfaces import JsonGenerator as ReceptJsonGen
from typing import Type
import time
from pydantic import BaseModel, Field


class GeminiJsonGenerator(
    GraphRagJsonGen, BoardJsonGen, ExpertSetJsonGen, RecovJsonGen, ReceptJsonGen
):
    def __init__(self, model: str = "gemini-2.0-flash-lite"):
        self.client = genai.Client(api_key="AIzaSyCYnrp56-_cToVfP4JWR1-t6SVG_K-in5c")
        self.model = model

    def generate_json(self, query: str, schema: Type[T]) -> T:
        # Gemini API may not support direct JSON schema enforcement, so we instruct it in the prompt
        prompt = f"""
        {query}
        Please respond in JSON format that matches the following schema:\n{schema.model_json_schema()}
        """
        for _ in range(80):  # Try up to 2 times
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": schema,
                    },
                )
                return schema.model_validate_json(response.text)
            except Exception as e:
                print(e)
                time.sleep(2)
        # Final attempt, let exception propagate if it fails again
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": schema,
            },
        )
        return schema.model_validate_json(response.text)


def example_usage():
    class GreetingModel(BaseModel):
        greeting: str = Field(..., description="A friendly greeting message.")

    # Instantiate the generator (replace with your actual API key)
    generator = GeminiJsonGenerator()

    # Define your query
    query = "Generate a greeting for a user."

    # Generate the JSON response
    result = generator.generate_json(query, GreetingModel)
    print(result)


if __name__ == "__main__":
    example_usage()
