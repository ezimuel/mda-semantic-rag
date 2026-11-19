# Master in Data Analytics - University of Roma Tre (Italy)
# Course: Semantic Search and RAG architectures
# by Enrico Zimuel (https://github.com/ezimuel)

import os
import json

from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers

# Load environment variables from .env file
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
env_path = os.path.join(parent_dir, ".env")

load_dotenv(dotenv_path=env_path)

input_path="data/scifi_movies_with_embeddings.jsonl"
if not os.path.isfile(input_path):
    raise FileNotFoundError(
        f"{input_path} not found. Execute 'python -m es.generate_embedding' first."
    )

# Connect to Elasticsearch
try:
    es = Elasticsearch(
        os.getenv("ELASTICSEARCH_URL"),
        basic_auth=(os.getenv("ELASTICSEARCH_USERNAME"), os.getenv("ELASTICSEARCH_PASSWORD"))
    )
except Exception as e:
    print(f"Error connecting to Elasticsearch: {e}")

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

successful_count = 0
error_count = 0
total = len(actions)

for processed, (ok, result) in enumerate(
    helpers.streaming_bulk(
        client=es,
        actions=actions,
        chunk_size=500,
        timeout="300s",
        refresh=True,
        raise_on_error=False,
    ),
    start=1,
):
    if ok:
        successful_count += 1
    else:
        error_count += 1

    if processed % 500 == 0 or processed == total:
        print(
            f"Processed {processed}/{total} documents "
            f"({successful_count} indexed, {error_count} errors)",
            flush=True,
        )
