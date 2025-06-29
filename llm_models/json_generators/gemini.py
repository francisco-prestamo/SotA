import os
from typing import Type
from google import genai
from graphrag.interfaces.json_generator import JsonGenerator as GraphRagJsonGen, T
from board.board import JsonGenerator as BoardJsonGen
from expert_set.interfaces import JsonGenerator as ExpertSetJsonGen
from recoverer_agent.interfaces import JsonGenerator as RecovJsonGen
from receptionist_agent.interfaces import JsonGenerator as ReceptJsonGen
from mocks.user_agent.interfaces import JsonGenerator as UserAgentJsonGen
import time
from pydantic import BaseModel, Field
from pydantic import ValidationError
from dotenv import load_dotenv

load_dotenv()


class GeminiJsonGenerator(
    GraphRagJsonGen,
    BoardJsonGen,
    ExpertSetJsonGen,
    RecovJsonGen,
    ReceptJsonGen,
    UserAgentJsonGen,
):
    def __init__(self, model: str = "gemini-2.0-flash-lite"):
        # List of API keys, can be loaded from env or passed directly
        
        self.api_keys = []
        self.api_keys = [os.getenv("GEMINI_API_KEY_1"), os.getenv("GEMINI_API_KEY_2"), os.getenv("GEMINI_API_KEY_3")]
        self.model = model
        self.key_index = 0
        self.client = genai.Client(api_key=self.api_keys[self.key_index])

    def rotate_key(self):

        self.key_index = (self.key_index + 1) % len(self.api_keys)
        self.client = genai.Client(api_key=self.api_keys[self.key_index])

    def generate_json(self, query: str, schema: Type[T]) -> T:
        prompt = f"""
        {query}
        Please respond in JSON format that matches the following schema:\n{schema.model_json_schema()}
        """
        max_schema_retries = 3
        schema_retries = 0
        while True:
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": schema,
                        "max_output_tokens": 100000,
                        "temperature": 0,
                    },
                )
                return schema.model_validate_json(response.text)
            except ValidationError as ve:
                schema_retries += 1
                print(f"Schema validation error, retrying ({schema_retries}/{max_schema_retries})...")
                if schema_retries >= max_schema_retries:
                    print("Schema validation failed after 5 attempts, returning empty JSON.")
                    prompt = "Generate an empty JSON response for each field in the schema."
                time.sleep(0.5)
                continue
            except Exception as e:
                # print("Rotating key...", e)
                self.rotate_key()
                time.sleep(10)
       


def example_usage():
    class GreetingModel(BaseModel):
        greeting: str = Field(..., description="A friendly greeting message.")

    # Instantiate the generator (replace with your actual API keys or set GEMINI_API_KEYS env)
    generator = GeminiJsonGenerator()

    # Define your query
    query = "Generate a greeting for a user."

    # Generate the JSON response
    result = generator.generate_json(query, GreetingModel)
    print(result)


if __name__ == "__main__":
    example_usage()
