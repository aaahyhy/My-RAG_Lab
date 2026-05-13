from src.ingestion.document_loader import load_documents
from src.ingestion.text_splitter import split_documents
from src.embeddings.embedding_model import get_embedding_model
from src.vectorstore.chroma_store import create_vectorstore
docs = load_documents("data/documents")
chunks = split_documents(docs)
embedding_model = get_embedding_model()
from src.embeddings.embedding_model import get_embedding_model

embedding_model = get_embedding_model()
test_vec = embedding_model.embed_query("test")
print("测试向量长度:", len(test_vec))
vectordb = create_vectorstore(
    chunks,
    embedding_model,
    "vector_db2"
)
results = vectordb.similarity_search(
    "What is value alignment in LLM?",
    k=3
)
for r in results:
    print(r.page_content[:200])
    print("metadata:", r.metadata)
    print()