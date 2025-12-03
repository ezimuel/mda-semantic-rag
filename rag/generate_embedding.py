# Master in Data Analytics - University of Roma Tre (Italy)
# Course: Semantic Search and RAG architectures
# by Enrico Zimuel (https://github.com/ezimuel)

import json
import ollama

input_path="data/scifi_movies.jsonl"
output_path="data/scifi_movies_with_embeddings.jsonl"

num_lines = 0
with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:

    for line in fin:
        num_lines += 1
        line = line.strip()
        if not line:
            continue

        doc = json.loads(line)
        response = ollama.embed(
            model="llama3.2:3b",
            input=doc["description"]
        )
        if response is None or "embeddings" not in response or len(response['embeddings']) == 0:
            print(f"Failed to generate embedding for document %d" % num_lines)
            continue

        print(f"Generated embedding for document %d" % num_lines)
        embedding = response['embeddings'][0]
        doc["embedding"] = embedding

        fout.write(json.dumps(doc) + "\n")

print(f"Processed {num_lines} lines, wrote documents with embeddings to {output_path}")
