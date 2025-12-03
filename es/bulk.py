# Master in Data Analytics - University of Roma Tre (Italy)
# Course: Semantic Search and RAG architectures
# by Enrico Zimuel (https://github.com/ezimuel)

import os
import json

from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers
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

input_path="data/scifi_movies.jsonl"
#input_path="data/scifi_movies_with_embeddings.jsonl"
action = {
    "_op_type": "index",
    "_index": "movies",
    "_id": 1
}
actions = []
with open(input_path, "r", encoding="utf-8") as fin:
    for line in fin:
        line = line.strip()
        doc = json.loads(line)
        actions.append(action | doc)
        action["_id"] += 1
       
successful_count, error_count = helpers.bulk(
    client=es,
    actions=actions,
    timeout='300s',
    refresh=True,
    raise_on_error=False,
    stats_only=True
)
print(f"Indexed {successful_count} documents with {error_count} errors.")