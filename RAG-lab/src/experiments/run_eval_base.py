import os
from src.vectorstore.chroma_store import load_vectorstore
from src.embeddings.embedding_model import get_embedding_model
from src.retrieval.base_retriever import BaseRetriever
from src.generation.llm_generator import LLMGenerator
from src.pipeline.rag_pipeline import RAGPipeline
from src.generation.llm_generator import get_llm
from src.evaluation.run_ragas import run_ragas
embedding_model = get_embedding_model()

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
db_path = os.path.join(project_root, "vector_db2")

print(f"正在尝试从该路径加载向量库: {db_path}")
# 3. 加载向量库并检查数量
vectordb = load_vectorstore(embedding_model, db_path)
print(f"DEBUG: 向量库中实际存储的条目数: {vectordb._collection.count()}")

retriever = BaseRetriever(vectordb)

generator = LLMGenerator()

pipeline = RAGPipeline(
    retriever=retriever,
    generator=generator,

)

llm = get_llm()
run_ragas(
    pipeline,
    "data/evaluation/ragas_testset.json",
    llm=llm,
    embeddings=embedding_model
)
