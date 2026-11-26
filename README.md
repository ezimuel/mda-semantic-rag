# MDA: Semantic Search & RAG architectures

This repository contains the source code of the course "Semantic Search & RAG architectures" for the [Master in Data Analytics](https://master-data-analytics.it/) from [University of Roma Tre](https://www.uniroma3.it/) in Italy.

In this course, we introduce the concept of semantic search and RAG architectures.

We collected some examples in Python using [Elasticsearch](https://github.com/elastic/elasticsearch) as data storage and
[Ollama](https://ollama.com/) as tool to run LLM locally.

We tested the examples using the embedding [qwen3-embedding:0.6b](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) and the LLM [Llama3.2 at 3B](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/).

The project contains the followig folders:
- `es` with the Elasticsearch examples of lexical search (term frequency), semantic search (vector search) and hybrid search.
- `rag` with examples of a RAG architecture using Elasticsearch and Ollama.

## Install the project examples

To install the Python examples, you can create and activate a virtual
environment ([venv](https://docs.python.org/3/library/venv.html)).

Use the following commands from the root folder of the repository:

```bash
python -m venv .venv
source .venv/bin/activate
```

After, you can install all the required packages as follows:

```bash
pip install -r requirements.txt
```

In order to execute the examples, you need to configure the `.env` file.
You can generate this file copying the `.env.dev` template file, as follows:

```bash
cp .env.dev .env
```

And then you can edit the `.env` file with your settings about Elasticsearch.

## Install Elasticsearch

To execute the examples reported in this repository you need to have an
instance of [Elasticsearch](https://www.elastic.co/elasticsearch) running. You can register for a free trial on
[Elastic Cloud](https://www.elastic.co/cloud/cloud-trial-overview) or install a local instance of Elasticsearch on your computer.

To install locally, you need to execute this command in the terminal:

```bash
curl -fsSL https://elastic.co/start-local | sh
```

This will install Elasticsearch and [Kibana](https://www.elastic.co/kibana) on macOS, Linux and Windows using WSL.

If you use Windows and you don't want to use WSL, you can install [Elasticsearch from a .zip file](https://www.elastic.co/docs/deploy-manage/deploy/self-managed/install-elasticsearch-with-zip-on-windows).

You can extract the `.zip` file in a folder and open it using the terminal (`cmd`).

Before executing Elasticsearch, you need to set the password for the `elastic` administrator user.
You can use the following command to set the password to `MyPassword` (choose the password that you want to use):

```shell
<nul set /p="MyPassword" | bin\elasticsearch-keystore.bat add -f -x bootstrap.password
```

After that, you can execute Elasticsearch with the following command to be executed in

```shell
bin\elasticsearch.bat -Expack.security.http.ssl.enabled=false
```

You need to set the `elastic`'s password in the `.env` file using the `ELASTICSEARCH_PASSWORD` key.

## Install Ollama

You can install Ollama using the following script for Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

For macOS or Windows you can follow the instructions reported [here](https://ollama.com/download).

When installed, you can run the Llama3.2 at 3b with the following command:

```bash
ollama run llama3.2:3b
```

This command will download the model (about 2 GB) and run a chat instance.
To exit from the chat interface you can use the `/bye` command.

## Generate the embeddings for the movies dataset

In this project we used the `qwen3-embedding:0.6b` embedding model.
You can download and install it using ollama, as follows:

```bash
ollama pull qwen3-embedding:0.6b 
```

The examples about semantic and hybrid search use the movies dataset stored in `data/scifi_movies.jsonl`. 
This file does not contains the vector embeddings. You can generate it using the script
`es/generate_embedding.py` as follows:

```bash
python -m es.generate_embedding
```

This script uses the `qwen3-embedding:0.6b` model to generate the embeddings
for the scifi movies reported in the dataset and store the results in a new `scifi_movies_with_embeddings.jsonl` file.

## Using Elasticsearch in Python

The `es` folder contains small examples using the Elasticsearch Python client.
Run the commands from the repository root, with the virtual environment activated
and Elasticsearch and Ollama running.

| Example | Description |
| --- | --- |
| `es/mapping.py` | Creates the `movies` index with movie metadata, nested actors, and a 1024-dimensional vector field using cosine similarity. |
| `es/embedding.py` | Generates an embedding for a sample sentence using Ollama and prints its dimensions and first values. |
| `es/generate_embedding.py` | Generates embeddings for movie descriptions and writes `data/scifi_movies_with_embeddings.jsonl`. |
| `es/bulk.py` | Loads the movies and their embeddings into Elasticsearch in batches, reporting indexing progress and errors. |
| `es/index.py` | Indexes a single example movie, without generating an embedding. |
| `es/get.py`, `es/update.py`, `es/delete.py` | Retrieve, update, or delete the movie with document ID `717`. |
| `es/search.py` | Performs lexical search on movie descriptions using a `match` query. |
| `es/fuzzy_search.py` | Demonstrates fuzzy matching with a misspelled search term. |
| `es/semantic_search.py` | Embeds a question and uses k-nearest-neighbor (kNN) vector search to find similar movie descriptions. |
| `es/hybrid_search.py` | Combines lexical and vector search using Reciprocal Rank Fusion (RRF). |
| `es/delete_index.py` | Deletes the entire `movies` index after asking for confirmation. |

To prepare the movies index, create its mapping, generate the embeddings
(if not already generated), and load the dataset:

```bash
python -m es.mapping
python -m es.generate_embedding
python -m es.bulk
```

Run `es.mapping` only when the `movies` index does not already exist.
Use the same `EMBEDDING_MODEL` for indexing and searching; its output must match
the mapping's 1024 dimensions.

You can then compare the search approaches:

```bash
python -m es.search
python -m es.fuzzy_search
python -m es.semantic_search
python -m es.hybrid_search
```

The queries and example document IDs are defined directly in these scripts.
The update and delete examples modify existing data, so use a course/demo index.

## Classic RAG

[`rag/rag.py`](rag/rag.py) demonstrates a fixed retrieval-augmented generation
pipeline using `data/nobel_prize_physics_2024.pdf`:

1. Extract the PDF text and split it into chunks of approximately 1000 characters.
2. Generate embeddings with Ollama and store the chunks in the separate
   `rag-example` index.
3. Embed the predefined question about the 2024 Nobel Prize in Physics and
   retrieve the five nearest chunks using Elasticsearch vector search.
4. Send the context and question to the Ollama language model and print its answer.

Set `EMBEDDING_MODEL=qwen3-embedding:0.6b` and `LLM_MODEL=llama3.2:3b` in `.env`,
and run:

```bash
python -m rag.rag
```

**Notes:** The script deletes and recreates `rag-example` on every run.
Currently, `context = text` replaces the retrieved chunks with the full PDF text
before generation. Remove that assignment to use only the retrieved context
in the final prompt.

## Agentic RAG

[`rag/agentic.py`](rag/agentic.py) uses Ollama's native tool calling to let the
language model choose how to query the existing `movies` index. Unlike the
fixed pipeline above, the agent can call tools, inspect their results, and make
another request before answering.

The example exposes two read-only tools:

- **`search_movies`** embeds a natural-language query and retrieves the five
  nearest movies using vector search, for recommendations and movie details.
- **`count_movies`** counts matching movies across the entire index, optionally
  filtering by an inclusive publication-year range and topic keywords. With
  `year_order="asc"` or `"desc"`, it also returns the oldest or latest movies,
  with a configurable `limit` (default 1, maximum 20).

Python executes the requested tools and sends their JSON results back to the
model. The loop stops when the model answers or after five model turns.

Prepare the movies index as described above, set `LLM_MODEL=llama3.2:3b` in
`.env`, and keep `EMBEDDING_MODEL` consistent with the indexed movie embeddings.
For example:

```bash
python -m rag.agentic "Suggest movies about space exploration"
python -m rag.agentic "How many movies are in the index?"
python -m rag.agentic "How many movies from 1980 are in the index?"
python -m rag.agentic "How many movies from 1970 to 1990 mention UFOs?"
python -m rag.agentic "What are the five latest movies in the index?"
```

Topic counts use lexical matching on titles and descriptions, not exhaustive
semantic matching. Oldest/latest refers to publication year, not indexing date;
movies published in the same year are tied.

## GraphRAG

[`rag/graph.py`](rag/graph.py) builds an in-memory NetworkX knowledge graph from
the existing `movies` index. It reads movie metadata (without embeddings), adds
cast/director links, and uses `LLM_MODEL` from `.env` to extract additional
entities and relationships from descriptions. Every relation retains its
source movie and evidence. The script only reads Elasticsearch.

Prepare the index as described above, install the requirements, and keep Ollama
running. The first run calls the LLM once per movie, so building the complete
graph can take a while:

```bash
python -m rag.graph --build-only
python -m rag.graph "Which movies share actors with Alien?"
python -m rag.graph "Who directed other movies starring actors from Alien?" --hops 3
```

The graph is saved as inspectable JSON in `data/movies_graph.json`. Later runs
load it without connecting to Elasticsearch or repeating extraction. Use
`--rebuild` after changing the dataset, extraction prompt, or model.
For a quick extraction demo, use a separate cache:

```bash
python -m rag.graph --limit 10 --graph data/demo_graph.json --build-only
```

A sample may not include Alien or enough connected movies for these questions.
`--limit` applies only when building; loading a cache always uses its stored graph.

Retrieval extracts entities from the question, matches normalized node names
(with conservative fuzzy matching), and traverses up to three hops in either
direction. For example, `Alien -> actor <- another movie -> director` supports
a multi-hop answer while preserving the direction of each original fact.
Retrieved facts are printed before the LLM generates an answer with source
citations. This example deliberately uses graph-only retrieval: no embedding
model, vector database extension, or GraphRAG framework is required.

For simplicity, traversal is capped at 100 nodes and 120 facts, with up to
eight question anchors. Answers are not exhaustive counts or lists. Extraction
can miss or misinterpret facts even when a supporting quote exists; the dataset
itself can also contain errors. Name normalization does not resolve aliases,
homonymous people/characters, or ambiguous references to movie remakes.
This is local graph traversal, not community-based GraphRAG summarization.

### Visualize the graph

Open [`rag/graph.html`](rag/graph.html) in a browser and choose
`data/movies_graph.json` (or a custom graph cache) using the file picker.
Alternatively, serve the repository root:

```bash
python -m http.server 8000 --bind 127.0.0.1
```

Visit <http://localhost:8000/rag/graph.html> to load the default cache automatically.
Scroll/pinch to zoom, drag to pan, and search to jump to an entity. Click a node
to see its attributes, movie metadata and connections; click a relation to see
its direction, source movies and evidence. Use **Fit graph** to reset the view
and toggle relation labels to reduce clutter.

The viewer is a single HTML file with no build step or backend dependencies.
It loads the pinned vis-network library from a CDN (internet required); graph
data stays in your browser. Elasticsearch and Ollama are not needed to view an
existing cache.

Here is reported a screenshot of the UI with a graph of 10 movies from the scifi archive.

![screenshot](./screenshot.png)

## Copyright

Copyright by [Enrico Zimuel](https://www.zimuel.it/), 2025.

## License
[![License: CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

The source code in this repository are licensed under the Creative Commons 
[Attribution–NonCommercial–ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en) License.
