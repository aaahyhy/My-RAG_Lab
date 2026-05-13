from src.vectorstore.chroma_store import load_vectorstore
from src.embeddings.embedding_model import get_embedding_model
from src.retrieval.base_retriever import BaseRetriever

embedding_model = get_embedding_model()

vectordb = load_vectorstore(
    embedding_model,
    "vector_db"
)

retriever = BaseRetriever(vectordb, k=3)

query = "What is value alignment in LLM?"

docs = retriever.retrieve(query)

for d in docs:

    print(d.page_content[:200])
    print("metadata:", d.metadata)
    print()