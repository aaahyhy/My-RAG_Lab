from typing import List
from langchain_core.documents import Document

class BaseRetriever:
    # 基础召回数默认设定为 3
    def __init__(self, vectorstore, k: int = 3):
        self.vectorstore = vectorstore
        self.k = k

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve top-k relevant documents
        """
        docs = self.vectorstore.similarity_search(
            query,
            k=self.k
        )
        return docs