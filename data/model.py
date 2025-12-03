# Master in Data Analytics - University of Roma Tre (Italy)
# Course: Semantic Search and RAG architectures
# by Enrico Zimuel (https://github.com/ezimuel)

from typing import List, Optional

from pydantic import BaseModel

class Actor(BaseModel):
    name: str
    characterName: Optional[str] = None
    name: str

class Movie(BaseModel):
    title: str
    description: str
    director: str
    datePublished: int
    aggregateRating: Optional[float] = None
    image: str
    url: str
    inLanguage: str
    actors: List[Actor]
    embedding: Optional[List[float]] = None