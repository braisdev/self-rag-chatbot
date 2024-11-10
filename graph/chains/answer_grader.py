from langchain_core.prompts import ChatPromptTemplate

from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class GradeAnswer(BaseModel):

    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


llm = ChatOpenAI(model_name="gpt-4o-mini")
structured_llm_grader = llm.with_structured_output(GradeAnswer, method="json_schema", strict=True)

system = """You are a grader assessing whether an answer addresses / resolves a question. \n
Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""

prompt_components = [
    ("system", system),
    ("human", "User question: \n\n {question} \n\n LLM generation: {generation}")
]

answer_grader_prompt = ChatPromptTemplate.from_messages(prompt_components)

answer_grader= answer_grader_prompt | structured_llm_grader
