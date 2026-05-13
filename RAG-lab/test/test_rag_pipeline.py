from src.vectorstore.chroma_store import load_vectorstore
from src.embeddings.embedding_model import get_embedding_model
from src.retrieval.base_retriever import BaseRetriever
from src.generation.llm_generator import LLMGenerator
from src.pipeline.rag_pipeline import RAGPipeline

embedding_model = get_embedding_model()

vectordb = load_vectorstore(
    embedding_model,
    "vector_db"
)
retriever = BaseRetriever(vectordb)
generator = LLMGenerator()
rag = RAGPipeline(
    retriever=retriever,
    generator=generator
)
query = "What is value alignment?"
answer = rag.run(query)
print(answer)