from sentence_transformers import CrossEncoder


class CrossEncoderReranker:

    def __init__(self):

        self.model = CrossEncoder(
            "BAAI/bge-reranker-base"
        )

    def rerank(self, query, docs, top_k=3):

        pairs = [
            (query, doc.page_content)
            for doc in docs
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [d[0] for d in ranked[:top_k]]