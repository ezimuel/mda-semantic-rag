# Master in Data Analytics - University of Roma Tre (Italy)
# Course: Semantic Search and RAG architectures
# by Enrico Zimuel (https://github.com/ezimuel)

import ollama

response = ollama.embed(
    model="llama3.2:3b",
    input="Spaceship flying in the outer space"
)
print(type(response))
print(f"Embeddings length: {len(response['embeddings'][0])}")
print(f"First 5 elements of embedding: {response['embeddings'][0][:5]}")