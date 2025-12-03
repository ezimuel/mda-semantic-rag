# MDA: Semantic Search & RAG architectures

This repository contains the source cod of the course "Semantic Search & RAG architectures" for the [Master in Data Analytics](https://master-data-analytics.it/) from [University of Roma Tre](https://www.uniroma3.it/) in Italy.

In this course, we introduce the concept of semantic search and RAG architectures.

We collected some examples in Python using [Elasticsearch](https://github.com/elastic/elasticsearch) as data storage and
[Ollama](https://ollama.com/) as tool to run LLM locally.

We tested the examples using [Llama3.2 at 3B](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/).

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

## Copyright

Copyright by [Enrico Zimuel](https://www.zimuel.it/), 2025.

## License
[![License: CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

The source code in this repository are licensed under the Creative Commons 
[Attribution–NonCommercial–ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en) License.



