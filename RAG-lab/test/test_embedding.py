from src.embeddings.embedding_model import get_embedding_model

embedding_model = get_embedding_model()

vector = embedding_model.embed_query("What is RAG?")

print("向量长度:", len(vector))
print("前5维:", vector[:5])