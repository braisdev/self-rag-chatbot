from dotenv import load_dotenv
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import initialize_retriever

from pprint import pprint
from graph.chains.generation import generation_chain
from langfuse.callback import CallbackHandler

load_dotenv()


def test_retrieval_grader_answer_yes() -> None:
    langfuse_handler = CallbackHandler(
        host="https://cloud.langfuse.com"
    )

    question = "Self Rag"
    retriever = initialize_retriever(
        persist_directory="../../../.chroma"
    )
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt},
        config={"callbacks": [langfuse_handler]},
    )

    assert res.binary_score == "yes"


def test_retrieval_grader_answer_no() -> None:
    langfuse_handler = CallbackHandler(
        host="https://cloud.langfuse.com"
    )
    question = "how to make pizza?"
    retriever = initialize_retriever(
        persist_directory="../../../.chroma"
    )
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt},
        config={"callbacks": [langfuse_handler]},
    )

    assert res.binary_score == "no"


def test_generation_chain() -> None:
    langfuse_handler = CallbackHandler(
        host="https://cloud.langfuse.com"
    )
    question = "Self Rag"
    retriever = initialize_retriever(
        persist_directory="../../../.chroma"
    )
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"context": docs, "question": question},
                                         config={"callbacks": [langfuse_handler]},)

    pprint(generation)
