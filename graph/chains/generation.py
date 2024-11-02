from langchain_core.prompts import ChatPromptTemplate


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
