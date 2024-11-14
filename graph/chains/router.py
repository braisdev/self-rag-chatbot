from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource"""

    # TIP: ... aka ellipsis means that the field would be required once we instantiate an object of this class

    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given an user question choose to route it to web search or vectorstore"
    )


llm = ChatOpenAI(model_name="gpt-4o-mini")
structured_llm_router = llm.with_structured_output(RouteQuery, method="json_schema", strict=True)

system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to self-rag, prompt engineering, and hallucinations in llms.
Use the vectorstore for questions on these topics. For all else, use web-search.
"""

messages = [
    ("system", system),
    ("human", "{question}"),
]

route_prompt = ChatPromptTemplate.from_messages(messages=messages)

question_router = route_prompt | structured_llm_router
