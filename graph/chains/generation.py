from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langfuse.callback import CallbackHandler
from dotenv import load_dotenv

load_dotenv()

# Create LLM instance from OpenAI
llm = ChatOpenAI(model_name="gpt-4o-mini")


prompt_components = [
    ("system", """
    You are an assistant for question-answering tasks. Use the provided context to answer the question.
    If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer 
    concise."""),
    ("human", """
    Question: {question}
    Context: {context}""")
]

prompt = ChatPromptTemplate.from_messages(prompt_components)

generation_chain = prompt | llm | StrOutputParser()
