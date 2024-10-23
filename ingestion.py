import os
import logging
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from typing import List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


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
            logger.info(f"Successfully loaded documents from {url}")
        except Exception as e:
            logger.error(f"Failed to load documents from {url}: {e}")
    return docs


def split_documents(
        documents: List[Document], chunk_size: int = 250, chunk_overlap: int = 0
) -> List[Document]:
    """
    Splits a list of Documents into smaller chunks using a recursive character text splitter.

    Args:
        documents: A list of Documents to split.
        chunk_size: The maximum number of characters in each chunk. Defaults to 250.
        chunk_overlap: The number of overlapping characters between chunks. Defaults to 0.

    Returns:
        A list of Documents, each representing a chunk of the original documents.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        logger.debug("Initialized RecursiveCharacterTextSplitter.")
    except Exception as e:
        logger.error(f"Failed to initialize text splitter: {e}")
        raise

    try:
        doc_splits = text_splitter.split_documents(documents=documents)
        logger.info(f"Split into {len(doc_splits)} document chunk(s).")

        for idx, doc in enumerate(doc_splits):
            source = doc.metadata.get('source', 'unknown')
            doc.id = f"{source}_{idx}"

        return doc_splits
    except Exception as e:
        logger.error(f"Error during document splitting: {e}")
        raise


def ingest_documents(
        documents: List[Document],
        persist_directory: str,
        collection_name: str = "rag_chroma"
) -> None:
    """
    Ingests documents into the Chroma vector store.

    Args:
        documents: A list of Documents to ingest.
        persist_directory: The directory where the vector store is persisted.
        collection_name: The name of the collection in the vector store.
    """
    embedding_function = OpenAIEmbeddings()

    try:
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=persist_directory,
            create_collection_if_not_exists=True
        )

        # Rest collection to handle new possible updates in the URLs and evade duplication
        vectorstore.reset_collection()
        logger.info(f"{collection_name} has been successfully reset.")

        ids = [doc.id for doc in documents]
        vectorstore.add_documents(documents, ids=ids)
        logger.info(f"Added {len(documents)} documents to the vector store.")
    except Exception as e:
        logger.error(f"Failed to add documents to vector store: {e}")
        raise


def main():
    """Main function to load environment variables and process documents."""
    logging.basicConfig(level=logging.INFO)
    load_dotenv()

    urls = [
        "https://selfrag.github.io/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination/"
    ]

    docs = load_documents(urls)
    split_docs = split_documents(docs)
    persist_directory = "./.chroma"
    ingest_documents(split_docs, persist_directory)


if __name__ == "__main__":
    main()
