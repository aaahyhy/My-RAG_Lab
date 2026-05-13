from src.ingestion.document_loader import load_documents

docs = load_documents("data/documents")
print("文档数量:", len(docs))
print("\n示例 Document:\n")
print(docs[0].page_content[:200])
print("\nmetadata:", docs[0].metadata)