from typing import Dict, Any

from langchain_core.documents import Document
from langchain_exa import ExaSearchResults

from dotenv import load_dotenv

from graph.state import GraphState

load_dotenv()

web_search_tool = ExaSearchResults()


def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    exa_results = web_search_tool.invoke({"query": question, "num_results": 3})

    joined_exa_results = "\n".join(
        [result.text for result in exa_results.results]
    )

    web_results = Document(page_content=joined_exa_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]

    print("fin")


if __name__ == "__main__":

    web_search(state={"question": "agent memory", "documents": None})



