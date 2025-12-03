# Master in Data Analytics - University of Roma Tre (Italy)
# Course: Semantic Search and RAG architectures
# by Enrico Zimuel (https://github.com/ezimuel)

import os
import json

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

movie = Movie(
    title="Forbidden Planet",
    description="A science fiction film about a spaceship crew that lands on a distant planet.",
    director="Fred M. Wilcox",
    datePublished=1956,
    aggregateRating=7.5,
    image="https://www.imdb.com/title/tt0049223/mediaviewer/rm1070205440/?ref_=tt_ov_i",
    url="https://www.imdb.com/title/tt0049223/",
    inLanguage="en",
    actors=[
        Actor(name="Leslie Nielsen", characterName="Commander John J. Adams"),
        Actor(name="Walter Pidgeon", characterName="Dr. Edward Morbius")
    ]
)

result = es.index(index="movies", document=movie.model_dump(exclude_none=True))
print(json.dumps(result.body, indent=4))