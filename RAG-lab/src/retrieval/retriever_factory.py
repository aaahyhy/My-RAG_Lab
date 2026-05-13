from src.retrieval.base_retriever import BaseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.bm25_retriever import BM25Retriever


def build_retriever(mode: str, vectordb):
    """
    Factory function to build different retrievers.

    Args:
        mode (str): retrieval mode
            - base
            - rerank
            - hyde
            - hybrid
        vectordb: vector database instance (Chroma)

    Returns:
        retriever instance
    """

    print(f"Building retriever mode: {mode}")

    # =========================
    # Dense Retriever
    # =========================

    if mode in ["base", "rerank", "hyde"]:

        retriever = BaseRetriever(vectordb)

        return retriever

    # =========================
    # Hybrid Retriever
    # =========================

    elif mode == "hybrid":

        # Dense retriever
        vector_retriever = BaseRetriever(vectordb)

        # 从向量库加载所有文档用于BM25
        print("Loading documents for BM25...")

        collection = vectordb._collection.get()

        documents = []

        for text in collection["documents"]:
            documents.append(text)

        # 创建BM25
        bm25_retriever = BM25Retriever(documents)

        # Hybrid
        retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever
        )

        return retriever

    # =========================
    # Unknown mode
    # =========================

    else:

        raise ValueError(f"Unknown retriever mode: {mode}")