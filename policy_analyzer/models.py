from pydantic import BaseModel
from typing import List, Optional


# Model representing a policy document
class Document(BaseModel):
    name: str
    content: str
    summary: Optional[str] = None
    source: Optional[str] = None


# Model representing a user question
class Question(BaseModel):
    query: str


# Model representing an AI-generated answer
class Answer(BaseModel):
    answer: str
    sources: List[str] = []
    confidence: Optional[float] = None 