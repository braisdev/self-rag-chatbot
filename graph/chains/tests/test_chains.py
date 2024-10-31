from dotenv import load_dotenv
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import initialize_retriever

load_dotenv()


def test_retrieval_grader_answer_yes() -> None:
    question = "Self Rag"
    retriever = initialize_retriever(
        persist_directory="../../../.chroma"
    )
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "yes"


def test_retrieval_grader_answer_no() -> None:
    question = "how to make pizza?"
    retriever = initialize_retriever(
        persist_directory="../../../.chroma"
    )
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "no"
