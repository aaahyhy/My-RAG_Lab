from langchain_community.vectorstores import Chroma
from typing import List
from langchain_core.documents import Document


def create_vectorstore(
    documents: List[Document],
    embedding_model,
    persist_dir: str
):

    vectordb = Chroma.from_documents(
        documents,
        embedding_model,
        persist_directory=persist_dir
    )

    vectordb.persist()

    return vectordb

def load_vectorstore(
    embedding_model,
    persist_dir: str
):

    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model
    )

    return vectordb