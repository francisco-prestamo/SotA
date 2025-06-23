import os
import time
from google import genai
from google.genai import types
from graphrag.interfaces.text_embedder import TextEmbedder
from entities.embedding import Embedding
import numpy as np
from dotenv import load_dotenv

load_dotenv()

class GeminiEmbedder(TextEmbedder):
	def __init__(self, dimensions: int = 10):
		self.dimensions = dimensions
		self.api_key = ""
		                
		self.client = genai.Client(api_key=self.api_key)


	@property
	def dim(self) -> int:
		return self.dimensions


	def embed(self, text: str) -> Embedding:
		while True:
			try:
				result = self.client.models.embed_content(
						model="text-embedding-004",
						contents=text,
						config=types.EmbedContentConfig(output_dimensionality=self.dimensions),
				)
				return Embedding(vector=np.array(result.embeddings[0].values))
			except Exception as e:
				print(e)
				time.sleep(2)
