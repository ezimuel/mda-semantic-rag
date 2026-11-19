# Master in Data Analytics - University of Roma Tre (Italy)
# Course: Semantic Search and RAG architectures
# by Enrico Zimuel (https://github.com/ezimuel)

import os
import ollama

from dotenv import load_dotenv

# Load environment variables from .env file
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
env_path = os.path.join(parent_dir, ".env")

load_dotenv(dotenv_path=env_path)

response = ollama.embed(
    model=os.getenv("EMBEDDING_MODEL"),
    input="Spaceship flying in the outer space"
)
print(type(response))
print(f"Embeddings length: {len(response['embeddings'][0])}")
print(f"First 5 elements of embedding: {response['embeddings'][0][:5]}")