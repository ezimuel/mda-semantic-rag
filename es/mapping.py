# Master in Data Analytics - University of Roma Tre (Italy)
# Course: Semantic Search and RAG architectures
# by Enrico Zimuel (https://github.com/ezimuel)

import os
import json

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

# Load environment variables from .env file
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
env_path = os.path.join(parent_dir, ".env")

load_dotenv(dotenv_path=env_path)

try:
    # Connect to Elasticsearch
    es = Elasticsearch(
        os.getenv("ELASTICSEARCH_URL"),
        basic_auth=(os.getenv("ELASTICSEARCH_USERNAME"), os.getenv("ELASTICSEARCH_PASSWORD"))
    )
except Exception as e:
    print(f"Error connecting to Elasticsearch: {e}")

# Create the movies index with mapping
index_name = "movies"
mappings = {
    "properties": {
        "title": {"type": "text"},
        "description": {"type": "text"},
        "director": {"type": "text"},
        "datePublished": {"type": "integer"},
        "aggregateRating": {"type": "float"},
        "image": {"type": "keyword"},
        "url": {"type": "keyword"},
        "inLanguage": {"type": "keyword"},
        "actors": {
            "type": "nested",
            "properties": {
                "name": {"type": "text"},
                "characterName": {"type": "text"}
            }
        },
        "embedding": {
            "type": "dense_vector", 
            "dims": 3072,
            "index": True,
            "similarity": "cosine"
        }
    }
}
result = es.indices.create(index=index_name, mappings=mappings)
print(json.dumps(result.body, indent=4))