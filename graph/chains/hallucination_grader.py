from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model_name="gpt-4o-mini")


class GradeHallucination(BaseModel):
    """Binary score for hallucination present in generation answer"""

    binary_score: bool = Field(
        description="Answer is grounded in facts, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeHallucination, method="json_schema", strict=True)

system = """You are a grader assessing whether an LLM Generation is grounded in / supported by a set of retrieved
facts. \nGive a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by a set of facts."""

prompt_components = [
    ("system", system),
    ("human", "Set of facts: \n\n {context} \n\n LLM generation: {generation}")
]

hallucination_prompt = ChatPromptTemplate.from_messages(prompt_components)

hallucination_grader = hallucination_prompt | structured_llm_grader
