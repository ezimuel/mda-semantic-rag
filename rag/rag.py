# Master in Data Analytics - University of Roma Tre (Italy)
# Course: Semantic Search and RAG architectures
# by Enrico Zimuel (https://github.com/ezimuel)

import re
import os
import json
import ollama
from ollama import Options
from pypdf import PdfReader
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

index_name = "rag-example"
model_name = "llama3.2:3b"

# Load environment variables from .env file
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
env_path = os.path.join(parent_dir, ".env")
pdf_path = os.path.join(parent_dir, "data/nobel_prize_physics_2024.pdf")

load_dotenv(dotenv_path=env_path)

# Connect to Elasticsearch
try:
    es = Elasticsearch(
        os.getenv("ELASTICSEARCH_URL"),
        basic_auth=(os.getenv("ELASTICSEARCH_USERNAME"), os.getenv("ELASTICSEARCH_PASSWORD"))
    )
except Exception as e:
    print(f"Error connecting to Elasticsearch: {e}")

# Load the PDF document
reader = PdfReader(pdf_path)
text = ""
print(f"Number of pages in PDF: {len(reader.pages)}")
for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text + "\n"

# Create the chunks from text using overlap
def split_text_by_chunks(text, chunk_size, overlap):
    cleantext = text.replace("\n", " ")
    sentences = re.split(r'(?<=[.!?])\s+', cleantext)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        # If adding this sentence exceeds chunk_size, finalize the current chunk
        if current_chunk and (current_length + sentence_length > chunk_size):
            chunks.append(" ".join(current_chunk))

            # Start new chunk with overlap from the previous chunk
            overlap_text = " ".join(current_chunk)[-overlap:] if overlap > 0 else ""
            current_chunk = [overlap_text.strip()] if overlap_text else []
            current_length = len(overlap_text)

        # Add sentence to the current chunk (even if it exceeds chunk_size)
        current_chunk.append(sentence)
        current_length += sentence_length

    # Add the last chunk if there is remaining text
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

chunks = split_text_by_chunks(text, 1000, 0)

# Create the index in Elasticsearch with dense_vector type

if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)

# Defining the mapping
response = es.indices.create(
    index=index_name, 
    mappings={
        "properties": {
            "embedding": {
                "type": "dense_vector",
                "dims": 3072
            },
            "my_text" : {
                "type" : "text"
            },
            "pdf_file" : {
                "type" : "keyword"
            }
        }
    }   
)
print("Index creation response:\n")
print(json.dumps(response.body, indent=4))

# Index the chunk in Elasticsearch
for i, chunk in enumerate(chunks):
    print(f"Indexing chunk: {i}\n")
    response = ollama.embed(model=model_name, input=chunk)
    embedding = response["embeddings"][0]
   
    response = es.index(
        index=index_name,
        document= {
            "embedding" : embedding,
            "my_text" : chunk,
            "pdf_file" : str(pdf_path)
        }
    )
    print(response)
    

question = "Chi ha vinto il premio nobel nel 2024?"
print(f"\nQuestion: {question}")

response = ollama.embed(model=model_name, input=question)
question_embedding = response["embeddings"][0]

# Semantic search: retrieval the relevant chunks
response = es.search(
    index=index_name,
    filter_path="hits.hits",
    knn={
        "field": "embedding",
        "query_vector": question_embedding,
        "k": 10,
        "num_candidates": 100
    },
    fields=[
        "my_text",
        "pdf_file"
    ],
    source=False
)

# Pass the chunks and the question to LLM
context = ""
for doc in response['hits']['hits']:
    context += doc["fields"]["my_text"][0]

context = text
prompt = (
    "Given the following context: \n"
    f"{context}\n"
    "Answer the following question:\n"
    f"{question}"
)

response = ollama.chat(
    model=model_name,
    messages=[{
        'role': 'user',
        'content': prompt
    }],
    options=Options(
        temperature=0.5
    )
)
print(response.message.content)