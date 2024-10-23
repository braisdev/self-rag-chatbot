from typing import List

from pydantic import BaseModel, Field


class GraphState(BaseModel):
    """
    Represents the state of the graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str = Field(
        description="question",
    )
    generation: str = Field(
        description="generation",
    )
    web_search: bool = Field(
        description="whether to add search",
    )
    documents: List[str] = Field(
        description="list of documents",
    )
