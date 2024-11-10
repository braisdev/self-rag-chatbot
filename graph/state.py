from typing import List
from xml.dom.minidom import DocumentType

from pydantic import BaseModel, Field
from langchain_core.documents import Document


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
        default="",
        description="question",
    )
    generation: str = Field(
        default="",
        description="generation",
    )
    web_search: bool = Field(
        default=False,
        description="whether to add search",
    )
    documents: List[Document] = Field(
        default_factory=list,
        description="list of documents",
    )
