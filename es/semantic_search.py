# Master in Data Analytics - University of Roma Tre (Italy)
# Course: Semantic Search and RAG architectures
# by Enrico Zimuel (https://github.com/ezimuel)

import os
import json
import ollama

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from data.model import Movie, Actor

# Load environment variables from .env file
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
env_path = os.path.join(parent_dir, ".env")

load_dotenv(dotenv_path=env_path)

# Connect to Elasticsearch
try:
    es = Elasticsearch(
        os.getenv("ELASTICSEARCH_URL"),
        basic_auth=(os.getenv("ELASTICSEARCH_USERNAME"), os.getenv("ELASTICSEARCH_PASSWORD"))
    )
except Exception as e:
    print(f"Error connecting to Elasticsearch: {e}")

response = ollama.embed(
    model="llama3.2:3b",
    input="An horror movie about a spaceship"
)
embedding = response['embeddings'][0]
print("Embedding vector length: %d\n" % len(embedding))

result = es.search(
    index="movies", 
    knn={
        "field": "embedding",
        "query_vector": embedding,
        "k": 5,
        "num_candidates": 50
    }
)
for hit in result.body['hits']['hits']:
    print(f"Found movie: {hit['_source']['title']} with score {hit['_score']} and id {hit['_id']}")

print(json.dumps(result.body, indent=4))