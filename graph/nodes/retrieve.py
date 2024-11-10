from typing import Any, Dict

from graph.state import GraphState
from ingestion import initialize_retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVER---")

    question = state.question

    retriever = initialize_retriever()
    documents = retriever.invoke(question)

    return {"documents": documents, "question": question}
