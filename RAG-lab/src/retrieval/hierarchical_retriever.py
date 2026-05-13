class HierarchicalRetriever:

    def __init__(self, summary_store, chunk_store):

        self.summary_store = summary_store
        self.chunk_store = chunk_store

    def retrieve(self, query):

        summary_docs = self.summary_store.similarity_search(
            query,
            k=3
        )

        pages = [
            d.metadata["page"]
            for d in summary_docs
        ]

        docs = []

        for p in pages:

            results = self.chunk_store.similarity_search(
                query,
                k=3,
                filter={"page": p}
            )

            docs.extend(results)

        return docs