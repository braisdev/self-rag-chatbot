import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(urls: List[str]) -> List[Document]:
    """
    Load documents from a list of URLs using Langchain's WebBaseLoader.

    Args:
        urls (List[str]): A list of URLs to load documents from.

    Returns:
        List[Document]: A flattened list of loaded documents.
    """
    docs = []
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            loaded_docs = loader.load()
            docs.extend(loaded_docs)
            logging.info(f"Successfully loaded documents from {url}")
        except Exception as e:
            logging.error(f"Failed to load documents from {url}: {e}")
    return docs


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents from a list of Documents.

    Args:
        documents (List[Document]): A list of Documents.

    Returns:
        List[Document]: A flattened list of loaded documents.
    """

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )

    doc_splits = text_splitter.split_documents(documents=documents)

    return doc_splits


def main():
    """Main function to load environment variables and documents."""
    load_dotenv()

    urls = [
        "https://selfrag.github.io/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination/"
    ]

    docs = load_documents(urls)

    split_docs = split_documents(docs)

    return docs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
