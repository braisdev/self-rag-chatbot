from dotenv import load_dotenv
from graph.graph import app
from langfuse.callback import CallbackHandler
load_dotenv()

if __name__ == "__main__":
    print("Hello Advanced Rag")

    langfuse_handler = CallbackHandler(
        host="https://cloud.langfuse.com"
    )

    print(app.invoke(input={"question": "steps to make an omelette"},
                     config={"callbacks": [langfuse_handler]}))
