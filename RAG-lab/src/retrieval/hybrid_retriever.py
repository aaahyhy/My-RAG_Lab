from collections import defaultdict


class HybridRetriever:
    """
    Hybrid Retriever using:
    - Dense Vector Retrieval
    - BM25 Keyword Retrieval

    Fusion method:
    Reciprocal Rank Fusion (RRF)
    """

    def __init__(
        self,
        vector_retriever,
        bm25_retriever,
        k=5,
        rrf_k=60
    ):
        """
        Args:
            vector_retriever: Dense retriever (BaseRetriever)
            bm25_retriever: BM25 retriever
            k: number of final documents
            rrf_k: RRF smoothing parameter
        """

        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.k = k
        self.rrf_k = rrf_k

    def retrieve(self, query):
        """
        Run hybrid retrieval.

        Steps:
        1. Dense retrieval
        2. BM25 retrieval
        3. RRF fusion
        """

        dense_docs = self.vector_retriever.retrieve(query, self.k)
        bm25_docs = self.bm25_retriever.retrieve(query, self.k)

        fused_docs = self.rrf_fusion(dense_docs, bm25_docs)

        return fused_docs

    def rrf_fusion(self, dense_docs, bm25_docs):
        """
        Reciprocal Rank Fusion
        """

        scores = defaultdict(float)
        doc_map = {}

        # Dense ranking
        for rank, doc in enumerate(dense_docs):

            key = doc.page_content

            scores[key] += 1 / (self.rrf_k + rank + 1)

            doc_map[key] = doc

        # BM25 ranking
        for rank, doc in enumerate(bm25_docs):

            key = doc.page_content

            scores[key] += 1 / (self.rrf_k + rank + 1)

            doc_map[key] = doc

        # sort
        sorted_docs = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        for key, _ in sorted_docs[:self.k]:
            results.append(doc_map[key])

        return results