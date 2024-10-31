from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)


class GradeDocuments(BaseModel):
    """Binary Score for relevance check on retrieved documents"""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'",
    )


structured_llm_grader = llm.with_structured_output(GradeDocuments, method="json_schema", strict=True)

prompt_components = [
    (
        "system", """You are a grader assessing relevance of a retrieved document to a user question. 
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.

        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    ),
    (
        "human", """Retrieved document: {document}
        User question: {question}"""
    )
]

grade_prompt = ChatPromptTemplate.from_messages(
    prompt_components
)

retrieval_grader = grade_prompt | structured_llm_grader
