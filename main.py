import os

from pydantic import BaseModel, Field

from llm_models.fireworksapi import FireworksApi


key = os.getenv("API_KEY")
if key == None:
    raise Exception()

api = FireworksApi(key)
print(api.generate_text("hello there"))

class GreetingModel(BaseModel):
    greeting: str = Field(examples=["guten tag", "bon jour"])
    name: str = Field(examples=["Piotr", "Peter", "John"])


print(api.generate_json(
    f"Generate a greeting, represented by the following schema: {GreetingModel.model_json_schema()}", 
    GreetingModel
).model_dump_json(indent=2))

